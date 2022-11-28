# === Fixed === #
import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr
import scipy.stats as stats

# === Task-specific === #
from copy import deepcopy
from dataset import NPweightingDataSet
from model import EmbedNet
from utils import *
from trainer_deers_np import logging, train, validate, test

# ====== Random Seed Initialization ====== #
def seed_everything(seed = 3078):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


# ====== DEERS: Model Definition ====== #
class DeepAutoencoderThreeHiddenLayers(nn.Module):
    def __init__(self, input_dim, hidden_dims, code_dim, activation_func=nn.ReLU,
                 code_activation=True, dropout=False, dropout_rate=0.5):
        super(DeepAutoencoderThreeHiddenLayers, self).__init__()
        # Establish encoder
        modules = []
        modules.append(nn.Linear(input_dim, hidden_dims[0]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
            
        for input_size, output_size in zip(hidden_dims, hidden_dims[1:]):
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
                
        modules.append(nn.Linear(hidden_dims[-1], code_dim))
        if code_activation:
            modules.append(activation_func())
        self.encoder = nn.Sequential(*modules)
        
        # Establish decoder
        modules = []
        
        modules.append(nn.Linear(code_dim, hidden_dims[-1]))
        modules.append(activation_func())
        if dropout:
            modules.append(nn.Dropout(dropout_rate))
        
        for input_size, output_size in zip(hidden_dims[::-1], hidden_dims[-2::-1]):
            modules.append(nn.Linear(input_size, output_size))
            modules.append(activation_func())
            if dropout:
                modules.append(nn.Dropout(dropout_rate))
        modules.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*modules)
        
    def forward(self, x):
        x = self.encoder(x)
        code = x
        x = self.decoder(x)
        return code, x
    

class ForwardNetworkTwoHiddenLayers(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, activation_func=nn.ReLU,
                out_activation=None, dropout_rate=0.0):
        super(ForwardNetworkTwoHiddenLayers, self).__init__()
        
        self.layers = nn.Sequential(
             nn.Linear(input_dim, hidden_dim1),
             activation_func(),
             nn.Dropout(dropout_rate),
             nn.Linear(hidden_dim1, hidden_dim2),
             activation_func(),
             nn.Linear(hidden_dim2, 1))
        
        self.out_activation = out_activation
        
    
    def forward(self, x):
        if self.out_activation:
            return self.out_activation(self.layers(x))
        else:
            return self.layers(x)
        

class DEERS_Concat(torch.nn.Module):
    def __init__(self, drug_autoencoder, cell_line_autoencoder, np_score_embed_net, 
                 forward_network):
        super(DEERS_Concat, self).__init__()
        self.np_score_embed_net = np_score_embed_net
        self.drug_autoencoder = drug_autoencoder
        self.cell_line_autoencoder = cell_line_autoencoder
        self.forward_network = forward_network
        
    def forward(self, drug_features, cell_line_features, np_features):
        np_code = self.np_score_embed_net(np_features)
        drug_code, drug_reconstruction = self.drug_autoencoder(drug_features)
        cell_line_code, cell_line_reconstruction = self.cell_line_autoencoder(cell_line_features)
        #cell_line_code = cell_line_code * np_code
        
        # Concatenate codes without interactions
        x = torch.cat((drug_code, np_code, cell_line_code), axis=1)    ####
        return self.forward_network(x), drug_reconstruction, cell_line_reconstruction
    
class MergedLoss(nn.Module):
    def __init__(self, y_loss_weight=1., drug_reconstruction_loss_weight=0.1, cell_reconstruction_loss_weight=0.2):
        super(MergedLoss, self).__init__()
        self.y_loss_weight = y_loss_weight
        self.drug_reconstruction_loss_weight = drug_reconstruction_loss_weight
        self.cell_reconstruction_loss_weight = cell_reconstruction_loss_weight
        self.output_criterion = nn.MSELoss()
        self.reconstruction_criterion = nn.MSELoss()
    
    def forward(self, pred_y, drug_reconstruction, cell_reconstruction, drug_input, cell_input, true_y):
        output_loss = self.output_criterion(pred_y, true_y)
        drug_reconstruction_loss = self.reconstruction_criterion(drug_reconstruction, drug_input)
        cell_reconstruction_loss = self.reconstruction_criterion(cell_reconstruction, cell_input)
        
        return output_loss, drug_reconstruction_loss, cell_reconstruction_loss


# ====== Argument Parsing ====== #
parser = argparse.ArgumentParser()
parser.add_argument('--WORKDIR_PATH', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP/2_DrugTarget_Integration_for_Drug_Response_Prediction")
#parser.add_argument('--DATASET_PATH', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data')
parser.add_argument('--inputdir', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input')
parser.add_argument('--np_input_fpath', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/iterative_enrich_np_consensus.tsv')
parser.add_argument('--drug_input_fpath', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/drug_data.tsv')
parser.add_argument('--exprs_input_fpath', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/rna_input.tsv')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--model_name', type=str, default='deers_netgp')
parser.add_argument('--split_type', type=str, default='drug', choices=['cell','drug','both','mix'])
parser.add_argument('--response_type', type=str, default='drug', choices=['IC50', 'AUC'])

# === Train setting === #
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--display_step', type=int, default=400)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--testset_yes', type=bool, default=True)

args = parser.parse_args()

# === Model Setting === #
args.code_dim = 64
args.embed_hid_dim = [128, 128]

args.drug_hidden_dims = [128, 128]
args.cell_hidden_dims = [128, 128]
args.code_dropout = True
args.code_dropout_rate = 0.2

args.forward_net_hidden_dim1 = 256
args.forward_net_hidden_dim2 = 128
args.forward_net_drop_rate = 0.5
args.forward_net_out_act = None

args.code_interactions = False

args.y_loss_weight=1.
args.drug_reconstruction_loss_weight=0.2
args.cell_reconstruction_loss_weight=0.2

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print('torch version: ', torch.__version__)
print(device)


def experiment(name_var1, name_var2, var1, var2, args, dataset_partition, model, loss_fn, device):
    
    # === Optimizer === #
    optimizer = optim.Adam([
            {'params': model.parameters()},
            {'params': loss_fn.parameters()}
        ], lr=args.learning_rate, weight_decay=args.weight_decay)

    # === Scheduler === #
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
    # scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1, verbose=True)
    
    # ====== Cross Validation Best Performance Dict ====== #
    best_performances = {}
    best_performances['best_epoch'] = 0
    best_performances['best_train_loss'] = float('inf')
    best_performances['best_train_corr'] = 0.0
    best_performances['best_valid_loss'] = float('inf')
    best_performances['best_valid_corr'] = 0.0
    # ==================================================== #
    
    list_epoch = []
    list_train_epoch_loss = []
    list_epoch_rmse = []
    list_epoch_corr = []
    list_epoch_spearman = []
    list_epoch_ci = []

    list_val_epoch_loss = []
    list_val_epoch_rmse = []
    list_val_epoch_corr = []
    list_val_spearman = []
    list_val_ci = []
    
    counter = 0
    for epoch in range(args.epochs):
        list_epoch.append(epoch)
        
        # ====== TRAIN Epoch ====== #
        model, list_train_batch_loss, list_train_batch_out, list_train_batch_true = train(model, epoch, train_loader, optimizer, loss_fn, device, args.display_step)
        
        epoch_train_rmse = np.sqrt(mean_squared_error(np.array(list_train_batch_out).squeeze(1), np.array(list_train_batch_true).squeeze(1)))
        epoch_train_corr, _p = pearsonr(np.array(list_train_batch_out).squeeze(1), np.array(list_train_batch_true).squeeze(1))
        epoch_train_spearman, _p = spearmanr(np.array(list_train_batch_out).squeeze(1), np.array(list_train_batch_true).squeeze(1))
        epoch_train_ci = concordance_index(np.array(list_train_batch_out).squeeze(1), np.array(list_train_batch_true).squeeze(1))
        
        train_epoch_loss = sum(list_train_batch_loss) / len(list_train_batch_loss)
        list_train_epoch_loss.append(train_epoch_loss)
        list_epoch_rmse.append(epoch_train_rmse)
        list_epoch_corr.append(epoch_train_corr)
        list_epoch_spearman.append(epoch_train_spearman)
        list_epoch_ci.append(epoch_train_ci)
        
        # ====== VALID Epoch ====== #
        list_val_batch_loss, list_val_batch_out, list_val_batch_true = validate(model, valid_loader, loss_fn, device)
        
        epoch_val_rmse = np.sqrt(mean_squared_error(np.array(list_val_batch_out).squeeze(1), np.array(list_val_batch_true).squeeze(1)))
        epoch_val_corr, _p = pearsonr(np.array(list_val_batch_out).squeeze(1), np.array(list_val_batch_true).squeeze(1))
        epoch_val_spearman, _p = spearmanr(np.array(list_val_batch_out).squeeze(1), np.array(list_val_batch_true).squeeze(1))
        epoch_val_ci = concordance_index(np.array(list_val_batch_out).squeeze(1), np.array(list_val_batch_true).squeeze(1))
        
        val_epoch_loss = sum(list_val_batch_loss)/len(list_val_batch_loss)
        list_val_epoch_loss.append(val_epoch_loss)
        list_val_epoch_rmse.append(epoch_val_rmse)
        list_val_epoch_corr.append(epoch_val_corr)
        list_val_spearman.append(epoch_val_spearman)
        list_val_ci.append(epoch_val_ci)
        
        if val_epoch_loss < best_performances['best_valid_loss']:
            best_performances['best_epoch'] = epoch
            best_performances['best_train_loss'] = train_epoch_loss
            best_performances['best_train_corr'] = epoch_train_corr
            best_performances['best_valid_loss'] = val_epoch_loss
            best_performances['best_valid_corr'] = epoch_val_corr
            torch.save(model, os.path.join(args.outdir, args.exp_name + f'_{name_var1}{var1}_{name_var2}{var2}.model'))
            model_max = deepcopy(model)
            
            counter = 0
        else:
            counter += 1
            logging(f'Early Stopping counter: {counter} out of {args.patience}', args.outdir, args.exp_name+'.log')
        
        logging(f'Epoch: {epoch:02d}, Train loss: {list_train_epoch_loss[-1]:.4f}, rmse: {epoch_train_rmse:.4f}, corr: {epoch_train_corr:.4f}, Valid loss: {list_val_epoch_loss[-1]:.4f}, rmse: {epoch_val_rmse:.4f}, pcc: {epoch_val_corr:.4f}', args.outdir, args.exp_name+'.log')
        if counter == args.patience:
            break
        
        scheduler.step(list_val_epoch_loss[-1])
        
    if args.testset_yes:
        test_loss, test_rmse, test_corr, test_spearman, test_ci, list_test_loss, list_test_out, list_test_true = test(model_max, test_loader, loss_fn, device)
        logging(f"Test:\tLoss: {test_loss}\tRMSE: {test_rmse}\tCORR: {test_corr}\tSPEARMAN: {test_spearman}\tCI: {test_ci}", args.outdir, f'{args.exp_name}_test.log')

        response_df = test_set.response_df
        response_df['test_loss'] = list_test_loss
        response_df['test_pred'] = list_test_out
        response_df['test_true'] = list_test_true
        response_df = response_df.sort_values(by='test_loss', ascending=True)
        filename = os.path.join(args.outdir, f'{args.exp_name}_{name_var1}{var1}_{name_var2}{var2}_test.result')
        response_df.drop('AUC', axis='columns').to_csv(filename, sep='\t', header=True, index=False)
    
    # ====== Add Result to Dictionary ====== #
    result = {}
    result['train_losses'] = list_train_epoch_loss
    result['val_losses'] = list_val_epoch_loss
    result['train_accs'] = list_epoch_corr
    result['val_accs'] = list_val_epoch_corr
    result['train_acc'] = epoch_train_corr
    result['val_acc'] = epoch_val_corr
    if args.testset_yes:
        result['test_acc'] = test_corr
    
    filename = os.path.join(args.outdir, f'{args.exp_name}_{name_var1}{var1}_{name_var2}{var2}_best_performances.json')
    with open(filename, 'w') as f:
        json.dump(best_performances, f)
        
    return vars(args), result, best_performances, model_max

# ====== Experiment Variable ====== #
name_var1 = ''
list_var1 = ['']
name_var2 = 'cv'
list_var2 = [f'{cv:02d}' for cv in range(20)]

total_results = defaultdict(list)
best_best_epoch = 0
best_best_train_loss = 99.
best_best_train_metric = 0
best_best_valid_loss = 99.
best_best_valid_metric = 0
best_var1_value = ''
best_var2_value = ''
for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        args.exp_name = f'{args.model_name}_{args.split_type}'
        args.outdir = os.path.join(args.WORKDIR_PATH, 'Results', args.exp_name)
        createFolder(args.outdir)
        logging(str(args), args.outdir, args.exp_name+'.log')
        logging(f"# ====== {name_var1}: {var1}, {name_var2}: {var2} ====== #", args.outdir, args.exp_name+'.log')
        
        # =============== #
        # === Dataset === #
        # =============== #
        train_set = NPweightingDataSet(
                    response_fpath=os.path.join(args.inputdir, f'cv_{args.split_type}', f'train_response_by_{args.split_type}_cv{args.cv}.tsv'), 
                    drug_input=args.drug_input_fpath, 
                    exprs_input=args.exprs_input_fpath, 
                    np_input=args.np_input_fpath,
                    response_type=args.response_type)
        valid_set = NPweightingDataSet(
                    response_fpath=os.path.join(args.inputdir, f'cv_{args.split_type}', f'valid_response_by_{args.split_type}_cv{args.cv}.tsv'), 
                    drug_input=args.drug_input_fpath, 
                    exprs_input=args.exprs_input_fpath, 
                    np_input=args.np_input_fpath,
                    response_type=args.response_type)
        test_set  = NPweightingDataSet(
                    response_fpath=os.path.join(args.inputdir, f'cv_{args.split_type}', f'test_response_by_{args.split_type}_cv{args.cv}.tsv'), 
                    drug_input=args.drug_input_fpath, 
                    exprs_input=args.exprs_input_fpath, 
                    np_input=args.np_input_fpath,
                    response_type=args.response_type)
        
        # === input === #
        args.drug_in_dim = train_set.drug_fp_df.shape[1] - 2
        args.cell_in_dim = train_set.cell_exprs_df.shape[1] - 2
        args.np_score_dim = train_set.np_score_df.shape[1] - 2
        
        print("-----------TRAIN DATASET-----------")
        print("NUMBER OF DATA:", train_set.__len__())
        print("-----------VALID DATASET-----------")
        print("NUMBER OF DATA:", valid_set.__len__())
        print("-----------TEST  DATASET-----------")
        print("NUMBER OF DATA:", test_set.__len__())

        # === Data Set/Loader === #
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=8)
        
        dataset_partition = {
            'train_loader': train_loader,
            'valid_loader': valid_loader,
            'test_loader' : test_loader
        }

        # ============= #
        # === Model === #
        # ============= #
        np_score_embed_net = EmbedNet(fc_in_dim=args.np_score_dim,
                                  fc_hid_dim=args.embed_hid_dim, 
                                  embed_dim=args.code_dim,
                                  dropout=0.1)

        drug_autoencoder = DeepAutoencoderThreeHiddenLayers(input_dim=args.drug_in_dim, 
                                     hidden_dims=args.drug_hidden_dims, 
                                     code_dim=args.code_dim,
                                     code_activation=True, dropout=args.code_dropout, dropout_rate=args.code_dropout_rate)

        cell_autoencoder = DeepAutoencoderThreeHiddenLayers(input_dim=args.cell_in_dim, 
                                         hidden_dims=args.cell_hidden_dims, 
                                         code_dim=args.code_dim,
                                         code_activation=True, dropout=args.code_dropout, dropout_rate=args.code_dropout_rate)

        # Forward network
        net = ForwardNetworkTwoHiddenLayers(3 * args.code_dim, 
                                                args.forward_net_hidden_dim1,
                                                args.forward_net_hidden_dim2,
                                                out_activation=args.forward_net_out_act,
                                                dropout_rate=args.forward_net_drop_rate)
        # Make the model together
        model = DEERS_Concat(drug_autoencoder=drug_autoencoder, 
                             cell_line_autoencoder=cell_autoencoder, 
                             np_score_embed_net=np_score_embed_net,
                             forward_network=net).to(device)

        # === Loss === #
        loss_fn = MergedLoss(y_loss_weight=args.y_loss_weight,
                             drug_reconstruction_loss_weight=args.drug_reconstruction_loss_weight,
                             cell_reconstruction_loss_weight=args.cell_reconstruction_loss_weight)
        
        # =============== #
        # ===== Run ===== #
        # =============== #
        setting, result, best_performances, model_max = experiment(name_var1, name_var2, var1, var2, args, dataset_partition, model, loss_fn, device)
        save_exp_result(setting, result, args.outdir)
        
        if best_performances['best_valid_corr'] >= best_best_valid_metric:
            best_best_epoch = best_performances['best_epoch']
            best_best_train_loss = best_performances['best_train_loss']
            best_best_train_metric = best_performances['best_train_corr']
            best_best_valid_loss = best_performances['best_valid_loss']
            best_best_valid_metric = best_performances['best_valid_corr']
            best_var1_value = var1
            best_var2_value = var2
            best_setting = setting
            best_result = result
        
        total_results[name_var1].append(var1)
        total_results[name_var2].append(var2)
        total_results['best_epoch'].append(best_performances['best_epoch'])
        total_results['best_train_loss'].append(best_performances['best_train_loss'])
        total_results['best_train_corr'].append(best_performances['best_train_corr'])
        total_results['best_valid_loss'].append(best_performances['best_valid_loss'])
        total_results['best_valid_corr'].append(best_performances['best_valid_corr'])
        
print(f'Best Train Loss: {best_best_train_loss:.4f}')
print(f'Best Train Corr: {best_best_train_metric:.4f}')
print(f'Best Valid Loss: {best_best_valid_loss:.4f}')
print(f'Best Valid Corr: {best_best_valid_metric:.4f}')

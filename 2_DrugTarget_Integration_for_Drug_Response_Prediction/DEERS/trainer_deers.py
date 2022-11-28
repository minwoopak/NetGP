import time
import os
import torch
from torch import nn
import torch.nn.functional as F

    
def logging(msg, outdir, log_fpath):
    fpath = os.path.join(outdir, log_fpath)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)
    print(msg)


def train(model, epoch, train_loader, optimizer, loss_fn, device, display_step=100):
    # ====== Train ====== #
    list_train_batch_loss = []
    list_train_batch_out  = []
    list_train_batch_true = []
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        drug_input, cell_input = tuple(d.to(device) for d in data)
        true_y = target.unsqueeze(1).to(device)
        
        optimizer.zero_grad()
        pred_y, drug_reconstruction, cell_reconstruction = model(drug_input, cell_input)
        output_loss, drug_reconstruction_loss, cell_reconstruction_loss = loss_fn(pred_y, drug_reconstruction, cell_reconstruction, drug_input, cell_input, true_y)
        
        loss = loss_fn.y_loss_weight * output_loss + \
               loss_fn.drug_reconstruction_loss_weight * drug_reconstruction_loss + \
               loss_fn.cell_reconstruction_loss_weight * cell_reconstruction_loss
        
        loss.backward()
        optimizer.step()
        
        list_train_batch_out.extend(pred_y.detach().cpu().numpy())
        list_train_batch_true.extend(true_y.detach().cpu().numpy())
        
        list_train_batch_loss.append(loss.detach().cpu().numpy())
    
        if batch_idx % display_step == 0 and batch_idx !=0:
            print(f'Epoch: {epoch}, minibatch: {batch_idx}, TRAIN: loss: {loss:.4f}')
        
    return model, list_train_batch_loss, list_train_batch_out, list_train_batch_true


def validate(model, valid_loader, loss_fn, device):
    list_val_batch_loss = []
    list_val_batch_out  = []
    list_val_batch_true = []
    
    model.eval()
    for batch_idx, (data, target) in enumerate(valid_loader):
        drug_input, cell_input = tuple(d.to(device) for d in data)
        true_y = target.unsqueeze(1).to(device)
        pred_y, drug_reconstruction, cell_reconstruction = model(drug_input, cell_input)
        output_loss, drug_reconstruction_loss, cell_reconstruction_loss = loss_fn(pred_y, drug_reconstruction, cell_reconstruction, drug_input, cell_input, true_y)
        
        loss = loss_fn.y_loss_weight * output_loss + \
               loss_fn.drug_reconstruction_loss_weight * drug_reconstruction_loss + \
               loss_fn.cell_reconstruction_loss_weight * cell_reconstruction_loss

        list_val_batch_out.extend(pred_y.detach().cpu().numpy())
        list_val_batch_true.extend(true_y.detach().cpu().numpy())

        list_val_batch_loss.append(loss.detach().cpu().numpy())
    
    return list_val_batch_loss, list_val_batch_out, list_val_batch_true


import numpy as np
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr,spearmanr

def test(model, test_loader, loss_fn, device):
    with torch.no_grad():
        # ====== Test ====== #
        list_test_loss = []
        list_test_out  = []
        list_test_true = []

        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            drug_input, cell_input = tuple(d.to(device) for d in data)
            true_y = target.unsqueeze(1).to(device)
            pred_y, drug_reconstruction, cell_reconstruction = model(drug_input, cell_input)
            output_loss, drug_reconstruction_loss, cell_reconstruction_loss = loss_fn(pred_y, drug_reconstruction, cell_reconstruction, drug_input, cell_input, true_y)
            
            loss = loss_fn.y_loss_weight * output_loss + \
                   loss_fn.drug_reconstruction_loss_weight * drug_reconstruction_loss + \
                   loss_fn.cell_reconstruction_loss_weight * cell_reconstruction_loss

            list_test_out.append(pred_y.detach().cpu().numpy().item())
            list_test_true.append(true_y.detach().cpu().numpy().item())

            list_test_loss.append(loss.detach().cpu().numpy())
    
    test_loss= sum(list_test_loss)/len(list_test_loss)
    
    test_rmse = np.sqrt(mean_squared_error(np.array(list_test_out), np.array(list_test_true)))
    test_corr, _p = pearsonr(np.array(list_test_out), np.array(list_test_true))
    test_spearman, _p = spearmanr(np.array(list_test_out), np.array(list_test_true))
    test_ci = concordance_index(np.array(list_test_out), np.array(list_test_true))
    
    #print(f"Test: Loss: {test_loss:.4f}, RMSE; {test_rmse:.4f}, CORR: {test_corr:.4f}, SPEARMAN: {test_spearman:.4f}, CI: {test_ci:.4f}")
    return test_loss, test_rmse, test_corr, test_spearman, test_ci, list_test_loss, list_test_out, list_test_true
    

from collections import defaultdict
import pandas as pd
def inspect_weights(model, test_set, test_loader, loss_fn, device):
    with torch.no_grad():
        # ====== Test ====== #
        list_gene_weights  = []
        list_test_out = []
        list_test_true = []
        list_test_loss = []

        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader):
            input_x = tuple(d.to(device) for d in data)
            true_y = target.unsqueeze(1).to(device)
            pred_y = model(*input_x)
            loss = loss_fn(pred_y, true_y)

            drug_emb, gene_weights, _ = model.get_embedding(*input_x)

            list_test_out.extend(pred_y.detach().cpu().numpy())
            list_test_true.extend(true_y.detach().cpu().numpy())
            list_test_loss.append(loss.detach().cpu().numpy())

            list_gene_weights.extend(gene_weights.detach().cpu().numpy())

        #test_rmse = mean_squared_error(np.array(list_test_out).squeeze(1), np.array(list_test_true).squeeze(1))**0.5
        #test_corr, p = stats.pearsonr(np.array(list_test_out).squeeze(1), np.array(list_test_true).squeeze(1))

    response_df = test_set.response_df
    cell_exprs_df = test_set.cell_exprs_df
    gene_symbols = cell_exprs_df.columns[2:].to_list()

    np_result_df = defaultdict(list)
    for i, row in response_df.iterrows():
        np_result_df['cell_name'].append(row['cell_name'])
        np_result_df['cell_idx'].append(row['cell_idx'])
        np_result_df['drug_name'].append(row['drug_name'])
        np_result_df['drug_idx'].append(row['drug_idx'])
        np_result_df['IC50'].append(row['IC50'])
        np_result_df['test_loss'].append(list_test_loss[i])
        for g_idx, symbol in enumerate(gene_symbols):
            gene_weight = list_gene_weights[i][g_idx]
            np_result_df[symbol].append(gene_weight)

    np_result_df = pd.DataFrame(np_result_df)
    return np_result_df


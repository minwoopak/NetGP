import os
import random
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
from Step2_DataEncoding import DataEncoding
from Step3_model_netgp import DeepTTC_NetGP_Concat, logging

def seed_everything(seed = 1024):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

parser = argparse.ArgumentParser()
# args, _ = parser.parse_known_args()

parser.add_argument('--workdir', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP")
parser.add_argument('--datadir', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data')
parser.add_argument('--inputdir', type=str, default='/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input')
parser.add_argument('--model_setting', type=str, default='DeepTTA_netgp_concat')
parser.add_argument('--split_type', type=str, default='drug', choices=['cell','drug','both','mix'])
parser.add_argument('--response_type', type=str, default='drug', choices=['IC50', 'AUC'])
parser.add_argument('--device', type=str, default='0')

args = parser.parse_args()

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print('torch version: ', torch.__version__)
print(device)
args.exp_name = f'{args.model_setting}_{args.split_type}'
args.outdir = os.path.join(args.workdir, 'Results', args.exp_name)
createFolder(args.outdir)


# ====== Encode Drug (from SMILES) ====== #
import codecs
from subword_nmt.apply_bpe import BPE
def drug2emb_encoder(smile):
    # === Convert SMILES to tokens and mask (also pad with 0 if shorter than len 50) === #
    vocab_path = os.path.join(args.datadir, 'drug_codes_chembl_freq_1500.txt')
    sub_csv = pd.read_csv(os.path.join(args.datadir, 'subword_units_map_chembl_freq_1500.csv'))

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # word 2 index dict

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)

cv_list = [f'{cv:02d}' for cv in range(20)]

test_result_df = []
for cv in cv_list:
    print(f"# ======= CV: {cv} ======= #")
    rnafile = os.path.join(args.inputdir, 'rna_input.tsv')
    npfile  = os.path.join(args.inputdir, 'iterative_enrich_np_consensus.tsv')
    args.traindata_fpath = os.path.join(args.inputdir, f'cv_{args.split_type}', f'train_response_by_{args.split_type}_cv{cv}.tsv')
    args.validdata_fpath = os.path.join(args.inputdir, f'cv_{args.split_type}', f'valid_response_by_{args.split_type}_cv{cv}.tsv')
    args.testdata_fpath  = os.path.join(args.inputdir, f'cv_{args.split_type}', f'test_response_by_{args.split_type}_cv{cv}.tsv')
    args.gdsc_smiles_fpath = os.path.join(args.datadir, 'gdsc_drug_smiles_data.tsv')

    rnadata = pd.read_csv(rnafile, sep='\t', header=0)
    rnadata = rnadata.drop('cell_idx', axis='columns')

    # ====== Import Response Data of desired split type ====== #
    traindata = pd.read_csv(args.traindata_fpath, header=0, sep='\t')
    validdata = pd.read_csv(args.validdata_fpath, header=0, sep='\t')
    testdata = pd.read_csv(args.testdata_fpath, header=0, sep='\t')
    
    train_rnaid = list(traindata['cell_name'])
    valid_rnaid = list(validdata['cell_name'])
    test_rnaid = list(testdata['cell_name'])
    rnadata = rnadata.set_index('cell_name').T

    train_rnadata = rnadata[train_rnaid]
    valid_rnadata = rnadata[valid_rnaid]
    test_rnadata = rnadata[test_rnaid]

    train_rnadata = train_rnadata.T  # convert to Cell(row)-by-Gene(col)
    valid_rnadata = valid_rnadata.T  # convert to Cell(row)-by-Gene(col)
    test_rnadata = test_rnadata.T    # convert to Cell(row)-by-Gene(col)
    train_rnadata.index = range(train_rnadata.shape[0])  # reset_index()
    valid_rnadata.index = range(valid_rnadata.shape[0])  # reset_index()
    test_rnadata.index = range(test_rnadata.shape[0])  # reset_index()

    # ====== NP Score Data ====== #
    npdata = pd.read_csv(npfile, sep='\t', header=0)
    npdata = npdata.drop('drug_idx', axis='columns')

    train_drugid = list(traindata['drug_name'])
    valid_drugid = list(validdata['drug_name'])
    test_drugid = list(testdata['drug_name'])
    npdata = npdata.set_index('drug_name').T

    train_npdata = npdata[train_drugid]
    valid_npdata = npdata[valid_drugid]
    test_npdata = npdata[test_drugid]

    train_npdata = train_npdata.T  # convert to Cell(row)-by-Gene(col)
    valid_npdata = valid_npdata.T  # convert to Cell(row)-by-Gene(col)
    test_npdata = test_npdata.T    # convert to Cell(row)-by-Gene(col)
    train_npdata.index = range(train_npdata.shape[0])  # reset_index()
    valid_npdata.index = range(valid_npdata.shape[0])  # reset_index()
    test_npdata.index = range(test_npdata.shape[0])  # reset_index()

    # ====== Drug / Response Data ====== #
    drug_smiles = pd.read_csv(args.gdsc_smiles_fpath, sep='\t', header=0)
    drug_smiles = drug_smiles.drop_duplicates(subset=['drug_name'])

    drug_smiles1 = drug_smiles.query('drug_name in @traindata.drug_name')
    drug_smiles2 = drug_smiles.query('drug_name in @validdata.drug_name')
    drug_smiles3 = drug_smiles.query('drug_name in @testdata.drug_name')
    drug_smiles = drug_smiles1.append(drug_smiles2).append(drug_smiles3).drop_duplicates()
    drug2smiles = dict(zip(drug_smiles['drug_name'], drug_smiles['smiles']))
    traindata['smiles'] = traindata['drug_name'].map(drug2smiles)
    validdata['smiles'] = validdata['drug_name'].map(drug2smiles)
    testdata['smiles'] = testdata['drug_name'].map(drug2smiles)

    traindata = traindata.rename(columns={'cell_name':'COSMIC_ID', 'drug_name':'DRUG_ID', args.response_type:'Label'})
    validdata = validdata.rename(columns={'cell_name':'COSMIC_ID', 'drug_name':'DRUG_ID', args.response_type:'Label'})
    testdata = testdata.rename(columns={'cell_name':'COSMIC_ID', 'drug_name':'DRUG_ID', args.response_type:'Label'})

    # === Drug SMILES Encoding === #
    smile_encode = pd.Series(drug_smiles['smiles'].unique()).apply(drug2emb_encoder)
    uniq_smile_dict = dict(zip(drug_smiles['smiles'].unique(),smile_encode))

    traindata['drug_encoding'] = [uniq_smile_dict[i] for i in traindata['smiles']]
    validdata['drug_encoding'] = [uniq_smile_dict[i] for i in validdata['smiles']]
    testdata['drug_encoding'] = [uniq_smile_dict[i] for i in testdata['smiles']]

    traindata = traindata.reset_index()
    validdata = validdata.reset_index()
    testdata = testdata.reset_index()

    modeldir = args.outdir
    modelfile = modeldir + '/model.pt'
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    net = DeepTTC_NetGP_Concat(modeldir=modeldir, device=device)
    net.train(train_drug=traindata, train_rna=train_rnadata, train_np=train_npdata,
              val_drug=validdata, val_rna=valid_rnadata, val_np=valid_npdata)
    net.save_model()
    print("Model Saveed :{}".format(modelfile))

    y_label, y_pred, mse, rmse, pearson, p_val, spearman, s_p_val, CI = net.predict(testdata, test_rnadata, test_npdata)
    print("Test Result: \n")
    logging(f"CV: {cv}\tRMSE: {rmse:.4f}\tPCC: {pearson:.4f}\tSpearman: {spearman:.4f}\tCI: {CI:.4f}", args.outdir, f'{args.exp_name}_test.log')

    test_result_df.append({
        'CV': cv,
        'Split_type': args.split_type,
        'RMSE': rmse, 
        'PCC': pearson, 
        'Spearman': spearman, 
        'CI': CI
    })

test_result_df = pd.DataFrame(test_result_df)
filename = os.path.join(args.outdir, f'{args.exp_name}_cv.result')
test_result_df.to_csv(filename, sep='\t', header=True, index=False)


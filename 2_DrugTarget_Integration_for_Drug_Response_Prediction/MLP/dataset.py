import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class DrugCellPairDataSet(Dataset):
    def __init__(self, response_fpath='/data/project/minwoo/Drug_recommendation/Data/model_input/train_dti_pair_response.tsv', 
             drug_input='/data/project/minwoo/Drug_recommendation/Data/model_input/drug_fingerprints.tsv', 
             exprs_input='/data/project/minwoo/Drug_recommendation/Data/model_input/cellline_expression_profiles.tsv',
             response_type='IC50'):
        
        self.response_df = pd.read_csv(response_fpath, sep='\t', header=0)
        self.drug_fp_df = pd.read_csv(drug_input, sep='\t', header=0)
        self.cell_exprs_df = pd.read_csv(exprs_input, sep='\t', header=0)
        self.response_type = response_type
    
    def __len__(self):
        return self.response_df.shape[0]
    
    def __getitem__(self, index):
        response_sample = self.response_df.iloc[index]
        cell_idx = response_sample['cell_idx']
        drug_idx = response_sample['drug_idx']
        
        response_val = torch.tensor(response_sample[self.response_type]).float()
        cell_vec = torch.from_numpy(np.array(self.cell_exprs_df[self.cell_exprs_df['cell_idx'] == cell_idx].iloc[:,2:])).float().squeeze(0)
        drug_vec = torch.from_numpy(np.array(self.drug_fp_df[self.drug_fp_df['drug_idx'] == drug_idx].iloc[:,2:])).float().squeeze(0)
        
        return (drug_vec, cell_vec), response_val


class NPweightingDataSet(Dataset):
    def __init__(self,
             response_fpath, 
             drug_input, 
             exprs_input,
             np_input,
             response_type='IC50'):
        
        self.response_df = pd.read_csv(response_fpath, sep='\t', header=0)
        self.drug_fp_df = pd.read_csv(drug_input, sep='\t', header=0)
        self.cell_exprs_df = pd.read_csv(exprs_input, sep='\t', header=0)
        self.np_score_df = pd.read_csv(np_input, sep='\t', header=0)
        self.response_type = response_type
    
    def __len__(self):
        return self.response_df.shape[0]
    
    def __getitem__(self, index):
        response_sample = self.response_df.iloc[index]
        cell_idx = response_sample['cell_idx']
        drug_idx = response_sample['drug_idx']
        
        response_val = torch.tensor(response_sample[self.response_type]).float()
        cell_vec = torch.from_numpy(np.array(self.cell_exprs_df[self.cell_exprs_df['cell_idx'] == cell_idx].iloc[:,2:])).float().squeeze(0)
        drug_vec = torch.from_numpy(np.array(self.drug_fp_df[self.drug_fp_df['drug_idx'] == drug_idx].iloc[:,2:])).float().squeeze(0)
        # === +1 to np_scores (곱했을때 0이 없고 높은애들은 높게 weighting되도록) === #
        np_score = torch.from_numpy(np.array(self.np_score_df[self.np_score_df['drug_idx'] == drug_idx].iloc[:,2:])).float().squeeze(0)
        
        return (drug_vec, cell_vec, np_score), response_val


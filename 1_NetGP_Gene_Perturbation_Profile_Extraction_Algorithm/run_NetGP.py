# ====== Create Network Propagation Profiles ====== #
# ====== Iterative Enrichment - Consensus Genes as final Seeds ====== #
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
import argparse
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import gseapy as gp
from run_NP import main_propagation
from sklearn.preprocessing import StandardScaler
import time

parser = argparse.ArgumentParser()

parser.add_argument('--rna_fpath', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/rna_input.tsv")
parser.add_argument('--drug_fpath', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/drug_data.tsv")
parser.add_argument('--ppi_fpath', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP/Data/9606.protein.links.symbols.v11.5.txt")
parser.add_argument('--out_fpath', type=str, default="/data/project/minwoo/Drug_recommendation/NetGP/Data/model_input/iterative_enrich_np_consensus_test.out")

args = parser.parse_args()


# ====== Import & Pre-process Data ====== #
drug_data = pd.read_csv(args.drug_fpath, sep='\t', header=0)
drug_list = drug_data['drug_name'].to_list()
drug2idx = dict(zip(drug_data['drug_name'], drug_data['drug_idx']))

target_data = pd.read_csv('/data/project/minwoo/Drug_recommendation/NetGP/Data/drug_target_info.tsv', sep='\t', header=0)
ppi_net = pd.read_csv(args.ppi_fpath, sep='\t', header=0)
ppi_genes = pd.Series(ppi_net['source'].append(ppi_net['target']).unique())
target_info_df = target_data[target_data['gene_name'].isin(ppi_genes)]

exps_input = pd.read_csv(args.rna_fpath, sep='\t', header=0)
gene_symbols = exps_input.columns[2:].to_list()

# ======================= #
# ====== Run NetGP ====== #
# ======================= #

args.restart_prob = 0.2
args.combined_score_cutoff = 800

args.netgp_dir = '/data/project/minwoo/Drug_recommendation/NetGP/Data/network'
createFolder(args.netgp_dir)

input_nwk_dir = os.path.join(args.netgp_dir, f'drug_specific_subnet_score_th_{args.combined_score_cutoff}')
seed_dir = os.path.join(args.netgp_dir, f'seeds_target_genes')

args.constantWeight = 'True'
args.absoluteWeight = 'False'
args.addBidirectionEdge = 'True'
args.normalize = 'True'

############################################# TEST #############################################
# drug_list = drug_list[:5]
outdir = os.path.join(args.netgp_dir, 'iterative_enrich_np_test')
createFolder(outdir)
outdir_intermediate = os.path.join(args.netgp_dir, 'iterative_enrich_np_test', f'iterative_np_test')
createFolder(outdir_intermediate)
############################################# TEST #############################################


# === Kegg or Reactome? === #
gene_sets = ['Reactome_2013', 'Reactome_2015', 'Reactome_2016']
#gene_sets = ['KEGG_2013', 'KEGG_2015', 'KEGG_2016', 'KEGG_2019_Human', 'KEGG_2019_Mouse', 'KEGG_2021_Human']

np_result_df = {}
for i, drug in enumerate(drug_list):
    print(f"# ====== Drug: {drug} ====== #")
    top_gene_sets_list = []
    # =============================================================== #
    # ============ Initial NP (Seeds: Drug Target Genes) ============ #
    # =============================================================== #
    args.input_graphs =  os.path.join(input_nwk_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
    args.seed = os.path.join(seed_dir, f'{drug}_targets.seed') # seed list file
    np_result = main_propagation(args)
    # === Index Unification === #
    np_result = np_result.set_index(np_result['protein'])
    np_result = np_result.reindex(gene_symbols).fillna(0)
    assert np_result.shape[0] == len(gene_symbols), f'np scores list not length: {len(gene_symbols)} but {np_result.shape[0]}'
    
    # === gene-by-drug np score result === #
    np_top_genes = np_result.sort_values(by='np_score', ascending=False).iloc[:200,:]['protein'].to_list()
    top_gene_sets_list.append(np_top_genes)
    
    # ====== Iteration Starts ====== #
    step = 1
    while True:
        print(f'# === Iteration {step} === #')
        step += 1
        # ============================== #
        # ========= Enrich Step ======== #
        # ============================== #
        try:
            enr = gp.enrichr(gene_list=np_top_genes,
                             gene_sets=gene_sets,
                             organism='human',
                             outdir=None
                            )
        except:
            time.sleep(10)
            enr = gp.enrichr(gene_list=np_top_genes,
                             gene_sets=gene_sets,
                             organism='human',
                             outdir=None
                            )

        sig_enr = enr.results.query('`Adjusted P-value` <= 0.05')

        sig_enr['Gene_list'] = sig_enr['Genes'].str.split(';').apply(lambda x: [g.strip() for g in x])
        sig_enr = sig_enr.explode('Gene_list')

        # === New Seed === #
        sig_path_genes = list(sig_enr['Gene_list'].unique())
        #top_gene_sets_list.append(sig_path_genes)

        seed_fpath = os.path.join(outdir_intermediate, f'{drug}_iterative_seed_genes.txt')
        pd.DataFrame(sig_path_genes).to_csv(seed_fpath, sep='\t', header=False, index=False)


        # ============================== #
        # =========== NP Step ========== #
        # ============================== #
        # === Iterative NP : w/ new seeds === #
        args.input_graphs =  os.path.join(input_nwk_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
        args.seed = seed_fpath
        np_result = main_propagation(args)

        np_result = np_result.set_index(np_result['protein'])
        np_result = np_result.reindex(gene_symbols).fillna(0)
        assert np_result.shape[0] == len(gene_symbols), f'np scores list not length: {len(gene_symbols)} but {np_result.shape[0]}'

        np_top_genes = np_result.sort_values(by='np_score', ascending=False).iloc[:200,:]['protein'].to_list()
        
        # === Converged? === #
        if set(np_top_genes) == set(top_gene_sets_list[-1]):
            break
        
        top_gene_sets_list.append(np_top_genes)
    
    # === Final NP : w/ consensus seeds === #
    print('# === Final Iteration === #')
    top_gene_sets_list = [set(g_lst) for g_lst in top_gene_sets_list]
    final_np_top_genes = set.intersection(*top_gene_sets_list)

    final_seed_fpath = os.path.join(outdir_intermediate, f'{drug}_final_seed_genes.txt')
    pd.DataFrame(final_np_top_genes).to_csv(final_seed_fpath, sep='\t', header=False, index=False)

    args.input_graphs =  os.path.join(input_nwk_dir, f'{drug}_subnet_score_{args.combined_score_cutoff}.nwk')
    args.seed = final_seed_fpath
    np_result = main_propagation(args)

    np_result = np_result.set_index(np_result['protein'])
    np_result = np_result.reindex(gene_symbols).fillna(0)
    assert np_result.shape[0] == len(gene_symbols), f'np scores list not length: {len(gene_symbols)} but {np_result.shape[0]}'
    
    # === gene-by-drug np score result === #
    np_result_df[drug] = np_result['np_score']
    
np_result_df = pd.DataFrame(np_result_df)
filename = os.path.join(outdir, 'iterative_enrich_np_test.out')
np_result_df.reset_index().to_csv(filename, sep='\t', header=True, index=False)


# ====== Keep Target Genes ====== #
#np_result_df = pd.read_csv(filename, sep='\t', header=0, index_col=0)
total_target_genes = pd.Series(target_info_df.query('drug_name in @drug_list')['gene_name'].unique())

mask = np_result_df.index.isin(total_target_genes)
np_result_filt = np_result_df[mask]

# === Standard Scaling === #
scaler = StandardScaler()
np_result_transform = scaler.fit_transform(np_result_filt)
np_result_transform = pd.DataFrame(np_result_transform, columns=np_result_filt.columns, index=np_result_filt.index)
np_result_transform = np_result_transform.transpose().reset_index().rename(columns={'index': 'drug_name'})
np_result_transform.insert(0, 'drug_idx', np_result_transform['drug_name'].map(drug2idx))

np_result_transform.to_csv(args.out_fpath, sep='\t', header=True, index=False)


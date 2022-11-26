#!/bin/bash

WORKDIR="/data/project/minwoo/Drug_recommendation/NetGP/1_NetGP_Gene_Perturbation_Profile_Extraction_Algorithm"
DATADIR="/data/project/minwoo/Drug_recommendation/NetGP/Data"

python run_NetGP.py \
    --rna_fpath $DATADIR//model_input/rna_input.tsv \
    --drug_fpath $DATADIR/model_input/drug_data.tsv \
    --ppi_fpath $DATADIR/9606.protein.links.symbols.v11.5.txt \
    --out_fpath $DATADIR/model_input/netGP_profile.out


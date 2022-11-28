#!/bin/bash

WORKDIR="/data/project/minwoo/Drug_recommendation/NetGP/2_DrugTarget_Integration_for_Drug_Response_Prediction/AGW/"

python train_agw.py --split_type both --response_type IC50 --device 0 #--epochs 1

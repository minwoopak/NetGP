#!/bin/bash

WORKDIR="/data/project/minwoo/Drug_recommendation/NetGP/"

python train_deeptta_netgp.py --split_type both --response_type IC50 --device 0
#!/bin/bash
# Script to run training on a single T4
export CUDA_VISIBLE_DEVICES=0

python3 aethelred_mini/train.py \
    --config aethelred_mini/configs/default.yaml \
    --device cuda

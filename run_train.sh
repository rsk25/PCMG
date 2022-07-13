#!/bin/bash

ENCODER_SIZE=$1
GPU_IDS=$3
ENCODER=google/electra-${ENCODER_SIZE}-discriminator
DATA=./resource/dataset/new_pen.json
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

SUBSET_TYPE=new_pen

EXPNAME=${SUBSET_TYPE}_${ENCODER_SIZE}_$2
EXPERIMENT=./resource/experiments/${SUBSET_TYPE}
EPOCH=500
WARMUP=10

echo -e "\033[33mExp name\033[0m: $EXPNAME"
echo -e "\033[33mData set\033[0m: $DATA"
echo -e "\033[33mEncoder \033[0m: $ENCODER"
echo -e "\033[33mEpoch   \033[0m: $EPOCH"
echo -e "\033[33mLearner \033[0m: COUNT=1; GPU=1"

killall -9 -r 'ray::'
python train_model.py -name $EXPNAME -data $DATA -exp $EXPERIMENT -model EPT-G -enc $ENCODER\
	-cpu 4 -gpu 1 -ids $GPU_IDS -iter $EPOCH -bsz 16 -lr 0.00176 -warmup $WARMUP\
	-cr 0 -da false

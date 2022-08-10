#!/bin/bash

ENCODER_SIZE=$1
SUBSET_TYPE=$2
EXPNAME=${SUBSET_TYPE}_${ENCODER_SIZE}_$3
EXPERIMENT=./resource/experiments/${SUBSET_TYPE}
DATA=./resource/dataset/${SUBSET_TYPE}.json
GPU_IDS=$4
ENCODER=google/electra-${ENCODER_SIZE}-discriminator

export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

EPOCH=$5
WARMUP=10
MODEL=$6

echo -e "\033[33mExp name\033[0m: $EXPNAME"
echo -e "\033[33mData set\033[0m: $DATA"
echo -e "\033[33mEncoder \033[0m: $ENCODER"
echo -e "\033[33mEpoch   \033[0m: $EPOCH"
echo -e "\033[33mLearner \033[0m: COUNT=1; GPU=1"


python train_model.py -name $EXPNAME -data $DATA -exp $EXPERIMENT -model $MODEL -enc $ENCODER\
	-cpu 4 -gpu 1 -ids $GPU_IDS -iter $EPOCH -bsz 16 -lr 0.00176 -warmup $WARMUP\
	-cr 0 -da false -fp 16

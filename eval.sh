#!/bin/bash


MODEL_NAME=./runs/$1
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATA=./resource/dataset/new_pen.json
SUBSET_TYPE=$2
EXPERIMENT=./resource/experiments/${SUBSET_TYPE}

python test_model.py -data $DATA -exp $EXPERIMENT -model ${MODEL_NAME}/best_* -ntr 500 -smp 50 -cpu 1 -gpu 0.5 
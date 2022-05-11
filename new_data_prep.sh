#!/bin/bash

DATA_NAME=$1
python preprocess.py -d ${DATA_NAME} --concat-to-pen --no-extra --fixed
python split_experiments.py

from pathlib import Path
import json
import random
import argparse

from experiment.dataset_spliter import *

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', '-d')
parser.add_argument('--experiment-path', '-e')

# DATA_PATH = Path('./resource/dataset/new_pen.json')
# EXP_PATH = Path('./resource/experiments/new_pen/')

if __name__ == "__main__":

    args = parser.parse_args()

    random.seed(9172)

    DATA_PATH = Path(f"./resource/dataset/{args.data_path}")
    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    assert type(dataset) == list

    id_list = get_ids(dataset)
    train_split, dev_split, test_split = split_data(id_list)

    EXP_PATH = Path(f"./resource/experiments/{args.experiment_path}")
    TRAIN = EXP_PATH / 'train'
    with TRAIN.open('w+t') as fp:
        for i in train_split:
            fp.write(i)
            fp.write('\n')

    DEV = EXP_PATH / 'dev'
    with DEV.open('w+t') as fp:
        for i in dev_split:
            fp.write(i)
            fp.write('\n')

    TEST = EXP_PATH / 'test'
    with TEST.open('w+t') as fp:
        for i in test_split:
            fp.write(i)
            fp.write('\n')

    print("Split Complete.")
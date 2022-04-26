from pathlib import Path
import json
import random

from experiment.dataset_spliter import *


DATA_PATH = Path('./resource/dataset/new_pen.json')
EXP_PATH = Path('./resource/experiments/pen/')


if __name__ == "__main__":

    random.seed(9172)

    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    assert type(dataset) == list

    id_list = get_ids(dataset)
    train_split, dev_split, test_split = split_data(id_list)

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
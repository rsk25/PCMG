from pathlib import Path
import json

from experiment.dataset_spliter import *


DATA_PATH = Path('./resource/dataset/new_pen.json')
EXP_PATH = Path('./experiments/pen/')


if __name__ == "__main__":

    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    assert type(dataset) == list
    
    train_index, dev_index, test_index = split_train_dev_test_indices(dataset)
    print(f"train_index: {train_index}")
    print(f"dev_index: {dev_index}")
    print(f"test_index: {test_index}")
    # train_split, dev_split, test_split = split_train_dev_test_id(dataset)


    # TRAIN = EXP_PATH / 'train'
    # with TRAIN.open('w+t') as fp:
    #     fp.writelines(train_split)

    # DEV = EXP_PATH / 'dev'
    # with DEV.open('w+t') as fp:
    #     fp.writelines(dev_split)

    # TEST = EXP_PATH / 'test'
    # with TEST.open('w+t') as fp:
    #     fp.writelines(test_split)
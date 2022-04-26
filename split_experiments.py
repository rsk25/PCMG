from pathlib import Path
import json

from experiment.dataset_spliter import *


DATA_PATH = Path('./resource/dataset/new_pen.json')
EXP_PATH = Path('./experiments/pen/')


if __name__ == "__main__":

    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    assert type(dataset) == list

    id_list = get_ids(dataset)
    print(id_list[:10])
    
    # train_indices, dev_indices, test_indices = set_indices(dataset)
    # train_split, _dataset = get_split_ids(dataset, train_indices)
    # dev_split, _dataset = get_split_ids(_dataset, dev_indices)
    # test_split, _dataset = get_split_ids(_dataset, test_indices)


    # TRAIN = EXP_PATH / 'train'
    # with TRAIN.open('w+t') as fp:
    #     fp.writelines(train_split)

    # DEV = EXP_PATH / 'dev'
    # with DEV.open('w+t') as fp:
    #     fp.writelines(dev_split)

    # TEST = EXP_PATH / 'test'
    # with TEST.open('w+t') as fp:
    #     fp.writelines(test_split)
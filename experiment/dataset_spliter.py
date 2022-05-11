from types import new_class
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

import random

from numpy import test


def get_ids(dataset: List[Dict[Any, Any]]) -> List[str]:
    id_list = []
    for problem in dataset:
        if problem.get('_exclude') == False or problem.get('dataset') == 'mawps':
            id_list.append(problem['_id'])


    return id_list


def split_data(id_list: List[str]) -> Tuple[List[int], List[int], List[int]]:
    total_num = len(id_list)
    random.shuffle(id_list)

    # set test indices
    test_split = id_list[:total_num//10]
    # set validation indices
    dev_split = id_list[total_num//10:total_num//10*2]
    # set train indices
    train_split = id_list[total_num//10*2:]
    assert len(train_split) + len(dev_split) + len(test_split) == total_num
    

    return train_split, dev_split, test_split


__all__ = ['get_ids','split_data']

from types import new_class
from typing import List, Dict, Any, Tuple
from pathlib import Path
import json

import numpy as np
from numpy.random import default_rng

random = default_rng(9172)


# split dataset using numpy into train/dev/test -> ratio: 8:1:1
    # must not overlap
    # choose without replacement
def split_train_dev_test_indices(dataset: List[Dict[Any, Any]]) -> Tuple[List[int], List[int], List[int]]:
    total_num = len(dataset)
    
    # get test indices
    indices = np.arange(total_num)
    test_indices = random.choice(indices, total_num//10, replace=False).tolist()
    # get validation indices
    indices = np.array([i for i in indices if i not in test_indices])
    dev_indices = random.choice(indices, total_num//10, replace=False).tolist()
    # get train indices
    train_indices = [i for i in indices if i not in dev_indices]
    # sanity check
    number_of_indices = len(test_indices) + len(dev_indices) + len(train_indices)
    assert number_of_indices == total_num, \
        f"total is {total_num}, but the number of indices is {number_of_indices}"

    
    return train_split, dev_split, test_split

# write split to non-extension text files
    # default save directory: /resource/experiments/pen/


__all__ = ['split_train_dev_test_indices']

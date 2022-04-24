from typing import List, Dict, Any
from pathlib import Path
import json

from numpy import arange
from numpy.random import default_rng

random = default_rng(9172)


# split dataset using numpy into train/dev/test -> ratio: 8:1:1
    # must not overlap
    # choose without replacement


# write split to non-extension text files
    # default save directory: /resource/experiments/pen/


__all__ = []

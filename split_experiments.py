from pathlib import Path
import json

from experiment.dataset_spliter import *

DATA_PATH = Path('/resource/dataset/new_pen.json')

if __name__ == "__main__":

    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    assert type(dataset) == list
    
    total_num = len(dataset)


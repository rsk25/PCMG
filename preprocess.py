from pathlib import Path
import re
from tqdm import tqdm
import argparse
import json

from preproc.data import Math23kDataset, Math23kProblem
from preproc.util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', '-d', required=True, default="math23k_only", type=str, help="The name of new dataset")
parser.add_argument('--concat-to-pen', action='store_true', \
    help="Concat new dataset to PEN; the output will be called 'new_pen.json'")
parser.add_argument('--no-concat', dest='concat-to-pen', action='store_false', \
    help="Will not concat new dataset to PEN")
parser.add_argument('--extra', action='store_true', \
    help="Will not save the file and instead get items that need extra preprocessing")
parser.add_argument('--no-extra', dest='extra', action='store_false', \
    help="Save the file and not get items that need extra preprocessing")
parser.add_argument('--fixed', action='store_true', \
    help="Will not toggle to debugging mode, when loading data")
parser.add_argument('--not-fixed', dest='fixed', action='store_false', \
    help="Toggle to debugging mode, when loading data")



if __name__ == '__main__':

    args = parser.parse_args()
    LOG_PATH = Path(f'./resource/dataset/')
    
    # Example of Math23kProblem class input (for single questions)
    # single_problem1 = Math23kProblem(
    #     oldText="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees. The children planted a tree every 2 meters (trees were planted at both ends of the road), and finally found that a total of 11 trees were planted. How many meters is the path long?",
    #     oldFormula=["x=(11-1)*2"],
    #     oldAnswer=["20"],
    #     mwp_template="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees . The children planted a tree every num3 meters ( trees were planted at both ends of the road ) , and finally found that a total of num1 trees were planted . How many meters is the path long ?",
    #     eqs_template=["x = ( num1 - num2 ) * num3"]
    # )
    
    # Loading the entire math23k dataset
    math23k = load_math23k(args.fixed)
    math23k_dataset = Math23kDataset()
    cnt = 0
    for problem in tqdm(math23k):
        cnt += 1
        math23k_problem = Math23kProblem(**problem)
        if args.extra:
            math23k_dataset.stack(math23k_problem)
        else:
            math23k_dataset.append_to_dataset(math23k_problem)

    if args.extra:
        list_of_buggy_probs = extra_preproc(math23k_dataset, r'million|billion', False)
        LOG_FILE = LOG_PATH / "buggy_probs.json"
        with LOG_FILE.open('w+t') as fp:
            json.dump(list_of_buggy_probs, fp)

    else:
        save_dataset(math23k_dataset.problems, args.dataset_name+'.json')

        if args.concat_to_pen:
            additional_data, number_of_excluded = non_excluded_only(math23k_dataset.problems)
            new_pen_dataset = concat_to_pen(additional_data)
            save_dataset(new_pen_dataset, 'new_pen.json')
            LOG_FILE = LOG_PATH / f"number_of_excluded_from_{args.dataset_name}"
            with LOG_FILE.open('w+t') as fp:
                fp.write(f"Number of excluded items: {number_of_excluded}")


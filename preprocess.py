from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm
import argparse

from preproc.data import Math23kDataset, Math23kProblem
from preproc.util import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', '-d', required=True, default="math23k_only", type=str, help="The name of new dataset")
parser.add_argument('--concat-to-pen', '-c', default=1, type=bool, help="Decides whether to concat to PEN or not; \
    the output will be called 'new_pen.json'")


if __name__ == '__main__':

    args = parser.parse_args()

    # Example of Math23kProblem class input (for single questions)
    # single_problem1 = Math23kProblem(
    #     oldText="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees. The children planted a tree every 2 meters (trees were planted at both ends of the road), and finally found that a total of 11 trees were planted. How many meters is the path long?",
    #     oldFormula=["x=(11-1)*2"],
    #     oldAnswer=["20"],
    #     mwp_template="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees . The children planted a tree every num3 meters ( trees were planted at both ends of the road ) , and finally found that a total of num1 trees were planted . How many meters is the path long ?",
    #     eqs_template=["x = ( num1 - num2 ) * num3"]
    # )
    
    # Loading the entire math23k dataset
    math23k = load_math23k()
    math23k_dataset = Math23kDataset()
    for problem in tqdm(math23k):
        math23k_problem = Math23kProblem(**problem)
        math23k_dataset.append_to_dataset(math23k_problem)

    save_dataset(math23k_dataset.problems, args.dataset_name+'.json')

    if args.concat_to_pen:
        new_pen_dataset = concat_to_pen(math23k_dataset.problems)
        save_dataset(new_pen_dataset, 'new_pen.json')


from pathlib import Path
from typing import List, Dict
import json
from tqdm import tqdm

from preproc.data import Math23kDataset, Math23kProblem
from preproc.util import *

if __name__ == '__main__':
    # single_problem1 = Math23kProblem(
    #     oldText="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees. The children planted a tree every 2 meters (trees were planted at both ends of the road), and finally found that a total of 11 trees were planted. How many meters is the path long?",
    #     oldFormula=["x=(11-1)*2"],
    #     oldAnswer=["20"],
    #     mwp_template="The children in the second grade of Zhenhai Yale School went to the side of a small road to plant trees . The children planted a tree every num3 meters ( trees were planted at both ends of the road ) , and finally found that a total of num1 trees were planted . How many meters is the path long ?",
    #     eqs_template=["x = ( num1 - num2 ) * num3"]
    # )
    # single_problem2 = Math23kProblem(
    #     oldText="Xiao Ming reads a story book. The first day he reads the whole book (1/6), the second day he reads 24 pages, and the third day reads 150% of the total number of pages read in the previous two days. At this time, there are still I haven't read (1/4) of the whole book, so how many pages does this book have?",
    #     oldFormula=["x=(24+24*150%)/(1-(1/6)-(1/6)*150%-(1/4))"],
    #     oldAnswer=["180"],
    #     mwp_template="Xiao Ming reads a story book . The first day he reads the whole book num4 , the second day he reads num1 pages , and the third day reads num2 of the total number of pages read in the previous 2 days . At this time , there are still I haven ' t read num5 of the whole book , so how many pages does this book have ?",
    #     eqs_template=["x = ( num1 + num1 * num2 ) / ( num3 - num4 - num4 * num2 - num5 )"]
    # )
    # print(single_problem1)
    # print(single_problem2.as_dict())

    math23k = load_math23k()
    math23k_dataset = Math23kDataset()
    for problem in tqdm(math23k):
        math23k_problem = Math23kProblem(**problem)
        math23k_dataset.append_to_dataset(math23k_problem)

    save_dataset(math23k_dataset.problems, 'math23k_only.json')

    new_pen_dataset = concat_to_pen(math23k_dataset.problems)
    save_dataset(new_pen_dataset, 'new_pen.json')


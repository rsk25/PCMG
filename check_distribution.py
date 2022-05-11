import json
from pathlib import Path
import argparse
import re

from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from edm import report


parser = argparse.ArgumentParser()
parser.add_argument('--with-report', action="store_true", help="Create report of classification difficulty")
parser.add_argument('--no-report', dest="with-report", action="store_false", help="No report")
parser.add_argument('--draw-graph', action="store_true", help="Draw graph")
parser.add_argument('--no-graph', dest='draw-graph', action="store_false", help="Don't draw graph")
parser.add_argument('--dataset', '-d', nargs='+', choices=['alg514','mawps','draw','math23k'], help="Choose dataset")
parser.add_argument('--compare-with-pen', action="store_true", help="Draw graph of pen for comparison")
parser.add_argument('--no-comparison', dest='compare-with-pen', action="store_false")


if __name__ == '__main__':

    args = parser.parse_args()

    DATA_PATH = Path("./resource/dataset/new_pen.json")
    
    with DATA_PATH.open('r+t') as fp:
        dataset = json.load(fp)
    
    text_data = []
    eqs_data = []
    for problem in dataset:
        if problem['dataset'] in args.dataset:
            text_data.append(problem['text'])
            equation = ''
            for eq in problem['equations']:
                equation += re.sub(r"[^+\-/\*()=]", "", eq)
            eqs_data.append(equation)


    ### Make report
    if args.with_report:
        result = report.get_difficulty_report(text_data, eqs_data)

        with Path(f'{args.dataset}_report.txt').open('w+t',encoding='utf-8') as fp:
            fp.write(result)
    
    
    ### Draw graph
    if args.draw_graph:
        count = Counter(eqs_data).most_common(100)
        df = pd.DataFrame(count, columns=['Category','Count'])
        
        sns.set(font_scale=0.2)
        sns.barplot(x=df['Category'], y=df['Count'])
        sns.lineplot(x=df['Category'], y=df['Count'])
        if not args.compare_with_pen:
            plt.xticks(rotation=90)
        else:
            plt.xticks()
        plt.savefig(f'{args.dataset}_graph.png', dpi=3000)

    ### Draw comparison
    if args.compare_with_pen:
        for problem in dataset:
            if problem['dataset'] in ['alg514','mawps','draw']:
                text_data.append(problem['text'])
                equation = ''
                for eq in problem['equations']:
                    equation += re.sub(r"[^+\-/\*()=]", "", eq)
                eqs_data.append(equation)
        
        count = Counter(eqs_data).most_common(100)
        df = pd.DataFrame(count, columns=['Category','Count'])
        sns.lineplot(x=df['Category'], y=df['Count'])
        
import itertools
from argparse import ArgumentParser

from scipy.stats import normaltest, pearsonr, spearmanr

from learner import *


def read_arguments():
    parser = ArgumentParser()
    parser.add_argument('--pickles', '-p', type=str, required=True, nargs='+')

    return parser.parse_args()


def star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '+'
    else:
        return ''


if __name__ == '__main__':
    args = read_arguments()
    set_seed(args.seed)

    metrics = {}
    keys = None
    for file in args.pickles:
        with Path(file).open('rb') as fp:
            id_to_metrics = pickle.load(fp)

        if keys is None:
            keys = sorted(id_to_metrics.keys())
        metrics[file.stem] = [id_to_metrics[k] for k in keys]

    for a, b in itertools.combinations(metrics.keys(), 2):
        print('-' * 80)
        data_a = metrics[a]
        data_b = metrics[b]

        # Check whether data follows normal
        _, p1 = normaltest(data_a)
        _, p2 = normaltest(data_b)
        print(f'H0: {a} is normal? p-value = {p1:.4f} {star(p1)}')
        print(f'H0: {b} is normal? p-value = {p2:.4f} {star(p2)}')

        if p1 > 0.05 and p2 > 0.05:
            test = 'Pearson\'s r'
            r, p = pearsonr(data_a, data_b)
        else:
            test = 'Spearman\'s r'
            r, p = spearmanr(data_a, data_b)
        print(f'Pearson\'s r = {r:.4f} (p-value = {p:.4f} {star(p)})')

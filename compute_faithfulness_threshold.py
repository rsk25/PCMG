from argparse import ArgumentParser
from numpy import mean

from common.data import Example
from common.dataset import Dataset
from experiment import base


def read_arguments():
    parser = ArgumentParser()
    parser.add_argument('--dataset', '-data', type=str, required=True)
    parser.add_argument('--experiment', '-exp', type=str, required=True, nargs='+')

    return parser.parse_args()


def threshold_compute(item: Example):
    tree = item.equation.to_tree_dict(item.info.variables)
    all_removed_tree = {
        'children': [],
        'name': '@'  # This name is not permitted in the original tree dict, so it ensures replacement effect
    }

    max_dist = base._compute_tree_edit_distance(tree, all_removed_tree)
    return 1.0 / float(max_dist)


if __name__ == '__main__':
    args = read_arguments()
    dataset = Dataset(args.dataset)

    for exp in args.experiment:
        dataset.select_items_with_file(exp)
        distances = [threshold_compute(item) for item in dataset.items]

        print('-' * 80)
        print(exp)
        print(mean(distances))

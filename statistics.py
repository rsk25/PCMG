from argparse import ArgumentParser
from pathlib import Path

import regex
from yaml import dump

from common.dataset import Dataset

PUNCT_END_REGEX = regex.compile('\\p{Punct}$')


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(curriculum=True, imitation=True, ignore_incorrect=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    env.add_argument('--experiment', '-exp', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = read_arguments()

    # Load raw data
    data = Dataset(args.dataset)
    print(dump(data.statistics))

    # Compute statistics
    for exp in Path(args.experiment).glob('*'):
        data.select_items_with_file(str(Path(exp).absolute()))
        print('-' * 80)
        print(exp)
        print(dump(data.get_statistics(False)))

    print('-' * 80)
    print('Total exp')
    data._items = sum(data._split_map.values(), [])
    print(dump(data.get_statistics(False)))


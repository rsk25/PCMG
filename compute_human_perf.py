from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List

import yaml

from common.data import Example, Explanation
from common.dataset import Dataset
from common.pen.metric import compute_metrics


def check(items: List[Example], tokenizer) -> dict:
    expl_references = {}
    expl_hypotheses = {}

    for item in items:
        if 'explanation_extra' not in item.info.raw:
            continue

        id_prefix = item.info.item_id + '::%s'
        expl_references.update({
            id_prefix % key: expl
            for key, expl in item.explanation.to_id_explanation_dict(tokenizer=tokenizer).items()
        })

        _hypo = Explanation.from_dict(item.info.raw, n_numbers=len(item.info.numbers),
                                      var_list=item.info.variables, tokenizer=tokenizer, field='explanation_extra')
        expl_hypotheses.update({
            id_prefix % key: expl
            for key, expl in _hypo.to_id_explanation_dict(tokenizer=tokenizer).items()
        })

    metrics = defaultdict(list)
    expl_keys = sorted(expl_references.keys())
    expl_references = {key: expl_references[key] for key in expl_keys}
    expl_hypotheses = {key: expl_hypotheses.get(key, ['-']) for key in expl_keys}

    for key, value in compute_metrics(expl_references, expl_hypotheses).items():
        metrics[key].append(value)

    return {key: float(sum(value) / len(value))
            for key, value in metrics.items()}


def read_arguments():
    parser = ArgumentParser()

    env = parser.add_argument_group('Dataset & Evaluation')
    env.set_defaults(curriculum=True, imitation=True, ignore_incorrect=True)
    env.add_argument('--dataset', '-data', type=str, required=True)
    env.add_argument('--experiment', '-exp', type=str, required=True, nargs='+')
    return parser.parse_args()


if __name__ == '__main__':
    args = read_arguments()

    dataset = Dataset(args.dataset)
    for exp in args.experiment:
        dataset.select_items_with_file(str(Path(exp).absolute()))
        print('-' * 80)
        print(exp)
        print(yaml.dump(check(dataset._items, dataset.tokenizer)))

    dataset._items = sum(dataset._split_map.values(), [])
    print('-' * 80)
    print('All')
    print(yaml.dump(check(dataset._items, dataset.tokenizer)))


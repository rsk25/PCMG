from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

DATA_PATH = Path('resource/dataset')
MATH23K_PATH = DATA_PATH / 'math23k_preproc'


def save_dataset(dataset, filename):
    file_path = DATA_PATH / filename
    with file_path.open('w+t') as fp:
        json.dump(dataset, fp)


def non_excluded_only(data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    new_data = []
    excluded_num = 0
    for d in data:
        if not d.get('_exclude', True):
            new_data.append(d)
        else:
            excluded_num += 1
    return new_data, excluded_num


def concat_to_pen(additional_data):
    assert type(additional_data) is list

    pen_file = DATA_PATH / 'pen.json'
    with pen_file.open('r+t') as fp:
        pen_data = json.load(fp)
    assert type(pen_data) is list
    new_pen_data = pen_data + additional_data
    return new_pen_data


def load_math23k():
    math23k = []
    math23k_file = MATH23K_PATH / 'math23k_translated.json'
    with math23k_file.open('r+t') as fp:
        for item in json.load(fp):
            math23k.append({
                'oldText': item['text_en'],
                'oldFormula': [item['equation']],
                'oldAnswer': [item['ans']]
            })

    def _update_math23k(template_name: str, file_obj, data_list):
        with file_obj.open('r+t') as fp:
            for i, item in enumerate(json.load(fp)):
                data_list[i].update({'%s_template' % template_name: item})

    for file in MATH23K_PATH.glob('*_template.json'):
        if 'eqs' in file.stem:
            _update_math23k('eqs', file, math23k)
        elif 'mwp' in file.stem:
            _update_math23k('mwp',file, math23k)
        else:
            raise FileNotFoundError()

    return math23k


__all__ = ['concat_to_pen','load_math23k','save_dataset','non_excluded_only']


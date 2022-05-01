import re
from typing import List, Dict, Any, Tuple
import json
from pathlib import Path

from sympy.core.numbers import E, _eval_is_eq

from .data import MathWordDataset

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


def load_math23k(fixed: bool):
    math23k = []
    math23k_file = MATH23K_PATH / 'math23k_translated.json'
    with math23k_file.open('r+t') as fp:
        for item in json.load(fp):
            math23k.append({
                'problem_id': item['id'],
                'oldText': item['text_en'],
                'oldFormula': [item['equation']],
                'oldAnswer': [item['ans']]
            })

    def _update_math23k(template_name: str, file_obj, data_list):
        with file_obj.open('r+t') as fp:
            for i, item in enumerate(json.load(fp)):
                data_list[i].update({'%s_template' % template_name: item})

    if fixed:
        for file in MATH23K_PATH.glob('*_template.json'):
            if 'eqs' in file.stem:
                _update_math23k('eqs', file, math23k)
            elif 'mwp' in file.stem:
                _update_math23k('mwp',file, math23k)
            else:
                raise FileNotFoundError()
    else:
        for file in DATA_PATH.glob('*_template.json'):
            if 'need_fix_eqs' in file.stem:
                _update_math23k('eqs', file, math23k)
            elif 'need_fix_mwp' in file.stem:
                _update_math23k('mwp',file, math23k)
            else:
                raise FileNotFoundError()

    return math23k


def extra_preproc(data_class: 'MathWordDataset', pattern: str, for_equation: bool) -> List[int]:
    list_of_buggy_probs = []
    problem_list = data_class.problems
    for problem in problem_list:
        if not problem._exclude:
            if for_equation:
                equation = problem.equations
                if re.search(pattern, equation[0]):
                    list_of_buggy_probs.append(
                        {
                            'id': problem.problem_id,
                            'orig_text': problem.oldText,
                            'preproc_eqs': equation,
                            'orig_equation': problem.oldFormula,
                            'orig_answer': problem.oldAnswer,
                            'orig_eqs_temp': problem.orig_eqs_template,
                            'orig_mwp_temp': problem.orig_mwp_template
                        }
                    )
            else:
                oldText = problem.oldText
                if re.search(pattern, oldText):
                    list_of_buggy_probs.append(
                        {
                            'id': problem.problem_id,
                            'orig_text': oldText,
                            'preproc_eqs': problem.equations,
                            'orig_equation': problem.oldFormula,
                            'orig_answer': problem.oldAnswer,
                            'orig_eqs_temp': problem.orig_eqs_template,
                            'orig_mwp_temp': problem.orig_mwp_template
                        }
                    )

    return list_of_buggy_probs


__all__ = [
    'concat_to_pen',
    'load_math23k',
    'save_dataset',
    'non_excluded_only',
    'extra_preproc'
]

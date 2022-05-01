import json
from re import M
from typing import List, Dict, Any
from pathlib import Path
from sympy.core.numbers import E
from tqdm import tqdm

from preproc.util import DATA_PATH, MATH23K_PATH

MWP_TEMPLATE_PATH = MATH23K_PATH / 'mwp_template.json'
EQS_TEMPLATE_PATH = MATH23K_PATH / 'eqs_template.json'
TEXT_PATH = MATH23K_PATH / 'math23k_translated.json'
BUGGY_PROBS_PATH = DATA_PATH / 'buggy_probs.json'


if __name__ == '__main__':
    
    with BUGGY_PROBS_PATH.open('r+t') as fp:
        buggy_data = json.load(fp)

    with TEXT_PATH.open('r+t') as fp:
        orig_dataset = json.load(fp)
    
    with MWP_TEMPLATE_PATH.open('r+t') as fp:
        mwp_templates = json.load(fp)
        assert type(mwp_templates) is list
    
    with EQS_TEMPLATE_PATH.open('r+t') as fp:
        eqs_templates = json.load(fp)
        assert type(eqs_templates) is list
    
    for data in tqdm(buggy_data):
        assert type(data) is dict
        problem_id = data.get('id') - 1 # id numbering in math23k starts with 1
        orig_dataset[problem_id]['text_en'] = data['orig_text']
        orig_dataset[problem_id]['equation'] = data['orig_equation'][0]
        orig_dataset[problem_id]['ans'] = data['orig_answer'][0]
        mwp_templates[problem_id] = data['orig_mwp_temp']
        eqs_templates[problem_id] = data['orig_eqs_temp']

    new_mwp = DATA_PATH / 'new_mwp_template.json'
    with new_mwp.open('w+t') as fp:
        json.dump(mwp_templates, fp, ensure_ascii=False)

    new_eqs = DATA_PATH / 'new_eqs_template.json'
    with new_eqs.open('w+t') as fp:
        json.dump(eqs_templates, fp, ensure_ascii=False)
    
    new_math23k = DATA_PATH / 'new_math23k_translated.json'
    with new_math23k.open('w+t') as fp:
        json.dump(orig_dataset, fp, ensure_ascii=False)

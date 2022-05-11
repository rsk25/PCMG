import yaml
import openpyxl
import re
from pathlib import Path


def read_equation(eqn):
    eqn = re.sub(' +', '', re.sub('R_\d{2}:', '\n', eqn)).strip().split('\n')
    equation = []
    expressions = {}
    variable = {}

    for rid, operation in enumerate(eqn):
        matched = re.match('^(.*)\((.*),(.*)\)', operation)
        opr = matched.group(1)

        if opr == '__NEW_VAR':
            expressions['R_%02d' % rid] = 'x_%d' % len(variable)
            variable['R_%02d' % rid] = 'x_%d' % len(variable)
        elif opr.startswith('__'):
            continue
        else:
            left = matched.group(2)
            right = matched.group(3)

            if left in expressions:
                left = '(%s)' % expressions.get(left)
            if right in expressions:
                right = '(%s)' % expressions.get(right)

            expressions['R_%02d' % rid] = '%s%s%s' % (left, opr, right)
            if opr == '=':
                equation.append('%s%s%s' % (left, opr, right))

    return '\n'.join(equation)


def read_explanation(expl):
    strings = []
    if 'dictitems' in expl:
        expl = expl['dictitems']
    for key, value in expl.items():
        strings.append(f'"{value[0]}" is {key}.')

    return '\n'.join(strings)


def dump_to_dict(items, name):
    mapping = {}
    for item in items['dump']:
        item_id = item['info']['item_id']
        gen_expl = read_explanation(item['expl_generated']) if 'expl_generated' in item else ''
        orig_expl = read_explanation(item['explanation'])
        gen_eq = read_equation(item['eqn_generated']['tokens'])
        orig_eq = read_equation(item['equation']['tokens'])

        mapping[item_id] = {
            f'{name}.expl': gen_expl,
            'gold.expl': orig_expl,
            f'{name}.equation': gen_eq,
            'gold.equation': orig_eq,
            f'{name}.correct': item['correct'],
            'text': item['text']['raw']
        }

    return mapping


with Path('./runs/seed_basic_base_ending_0907045246/FESTA-71811_00001/dev.yaml').open() as fp:
    festa = dump_to_dict(yaml.load(fp, Loader=yaml.Loader), 'FESTA')

with Path('./runs/seed_basic_base_last_0905063455/FESTA_noO-6264a_00002/dev.yaml').open() as fp:
    festa_noo = dump_to_dict(yaml.load(fp, Loader=yaml.Loader), 'FESTA_noO')

with Path('./runs/seed_basic_base_last_0905063455/FESTA_noR-6264a_00003/dev.yaml').open() as fp:
    festa_nor = dump_to_dict(yaml.load(fp, Loader=yaml.Loader), 'FESTA_noR')

with Path('./runs/seed_basic_base_last_0905063455/EPT-6264a_00000/dev.yaml').open() as fp:
    ept = dump_to_dict(yaml.load(fp, Loader=yaml.Loader), 'EPT')


workbook = openpyxl.Workbook()
sheet = workbook.create_sheet('AllCases')

row_keys = ['item_id', 'text', 'EPT.correct', 'FESTA.correct', 'FESTA_noO.correct', 'FESTA_noR.correct',
            'gold.expl', 'FESTA.expl', 'FESTA_noO.expl', 'FESTA_noR.expl',
            'gold.equation', 'EPT.equation', 'FESTA.equation', 'FESTA_noO.equation', 'FESTA_noR.equation']
sheet.append(row_keys)

for key in ept:
    item = ept[key].copy()
    item.update(festa[key])
    item.update(festa_noo[key])
    item.update(festa_nor[key])

    sheet.append([key] + [item[c] for c in row_keys[1:]])

workbook.save('error_analysis.xlsx')
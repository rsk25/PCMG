import json
from collections import defaultdict
from pathlib import Path

RSC_PATH = Path('resource')
DATA_PATH = RSC_PATH / 'dataset' / 'pen.json'

with DATA_PATH.open('rt') as fp:
    data = json.load(fp)

for item in data:
    expl = item['explanations']
    expl_test = item.get('explanation_extra', None)
    all_expl = list(expl.items()) + ([] if expl_test is None else list(expl_test.items()))
    key_to_expl = defaultdict(list)
    key_to_num = {n['key']: n['value'] for n in item['numbers']}

    for worker, workerset in all_expl:
        for key, desc in workerset.items():
            if desc == '':
                continue

            duplicates = {k for k, d in workerset.items() if k != key and d == desc}
            assert all(k.startswith('N_') for k in duplicates)
            assert all(key_to_num[k] == key_to_num[key] for k in duplicates), (duplicates, key_to_num, item['_id'])

        # for key, desc in workerset.items():
        #     if desc != '':
        #         key_to_expl[key].append(worker)

    # max_length = len(all_expl)
    # for key, works in key_to_expl.items():
    #     if len(works) == max_length:
    #         continue
    #
    #     # find duplicate position
    #     worked = [workerset for worker, workerset in all_expl if worker in works]
    #     not_worked = [workerset for worker, workerset in all_expl if worker not in works]
    #
    #     print('-' * 80)
    #     print('Text: %s' % item['text'])
    #     print('Target number: %s' % key)
    #     if key.startswith('N_'):
    #         print(item['numbers'][int(key[2:])])
    #
    #     print('Duplicates:')
    #     intersection = None
    #     for workerset in worked:
    #         duplicates = [k for k, desc in workerset.items() if k != key and desc == workerset[key]]
    #         print('A worker said %s are duplicates' % duplicates)
    #         if intersection is None:
    #             intersection = set(duplicates)
    #         else:
    #             intersection = intersection.intersection(duplicates)
    #
    #     if len(intersection):
    #         print('Intersection: %s' % intersection)
    #     else:
    #         print('There is no intersection.')
    #
    #     yn = input('Apply %s to duplicate other workers (A) or delete worked ones (d)?' % intersection).lower().strip()
    #     if yn == 'd':
    #         for wset in worked:
    #             wset[key] = ''
    #     else:
    #         for wset in not_worked:
    #             desc = ''
    #             for k in intersection:
    #                 if wset[k]:
    #                     desc = wset[k]
    #                     break
    #
    #             wset[key] = desc

# with DATA_PATH.open('w+t') as fp:
#     json.dump(data, fp, indent=2)



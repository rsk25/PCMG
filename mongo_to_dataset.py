from json import load, dump
from pathlib import Path

RSC_PATH = Path('resource')
DATA_PATH = RSC_PATH / 'dataset'


def clean_dash(item):
    if type(item) is dict:
        return {k: clean_dash(v) for k, v in item.items()}
    if type(item) is list:
        return [clean_dash(v) for v in item]
    if type(item) is str:
        item = item.strip()
        return item if item != '-' else ''
    return item


train_dataset = []
valid_dataset = []
for file in DATA_PATH.glob('*.json'):
    if file.stem != 'pen':
        with file.open('r+t') as fp:
            for item in load(fp):
                if item['_id'] == '604cc853117d71c492a62ea4':
                    continue
                if '_done' in item:
                    del item['_done']
                if '_reported' in item:
                    del item['_reported']
                del item['worker']

                item['equations'] = item['simpleEq']
                del item['simpleEq']

                explanations = clean_dash(item['descriptions'])
                del item['descriptions']

                if len(explanations)>3:
                    # Exclude one for human-level performance
                    item['explanation_extra'] = {'workerExtra': explanations[item['worker_for_train']]}
                    item['explanations'] = {key: value
                                            for key, value in explanations.items()
                                            if key != item['worker_for_train']}
                    item['worker_for_train'] = None
                    valid_dataset.append(item)
                else:
                    item['explanations'] = explanations
                    train_dataset.append(item)

with (DATA_PATH / 'pen.json').open('w+t') as fp:
    dump(train_dataset + valid_dataset, fp, indent=2)

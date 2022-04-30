# EPT-G: EPT-based Math Problem Generator
Inspired by EPT-X ([Kim et al., 2021](https://www.2022.aclweb.org/papers)) and [Wang et al. 2021](https://aclanthology.org/2021.emnlp-main.484/).

Math23k English translation provided by [Zichao (Jack) Wang](https://zw16.web.rice.edu) 王子超. (Github link: [https://github.com/moonlightlane](https://github.com/moonlightlane))

Original PEN dataset provided by [Bugeun Kim, Ph.D.](https://scholar.google.com/citations?user=6zDxUP8AAAAJ&hl=ko&oi=ao) 김부근. (Github link: [https://github.com/nearbydelta](https://github.com/nearbydelta))

# Data
We use the PEN and Math23k as training data.
## Preprocessing Math23k
```
python preprocess.py -d math23k_only --no-extra --fixed
```
Man page for `preprocess.py`:
```
usage: preprocess.py [-h] --dataset-name DATASET_NAME [--concat-to-pen CONCAT_TO_PEN] [--extra] [--no-extra] [--fixed] [--not-fixed]

optional arguments:
  -h, --help            show this help message and exit
  --dataset-name DATASET_NAME, -d DATASET_NAME
                        The name of new dataset
  --concat-to-pen CONCAT_TO_PEN, -c CONCAT_TO_PEN
                        Decides whether to concat to PEN or not; the output will be called 'new_pen.json'
  --extra               Will not save the file and instead get items that need extra preprocessing
  --no-extra            Will save the file and not get items that need extra preprocessing
  --fixed               Will not toggle to debugging mode, when loading data
  --not-fixed           Will toggle to debugging mode, when loading data
```
### Bug fixing
```
python preprocess.py -d math23k_only --extra --not-fixed
python fix_buggy_probs.py
```
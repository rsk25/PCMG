from pathlib import Path
from time import sleep
from typing import List, Dict, Set

from numpy import mean, std
from numpy.random import Generator, PCG64

from common.const.model import DEF_ENCODER
from common.const.operator import OPR_NEW_VAR_ID
from common.data import Example

CACHE_ITEMS = 'item'
CACHE_TOKENIZER = 'tokenizer'
CACHE_VOCAB_SZ = 'vocab_size'


def _word_counts(indices, tokenizer):
    single_dim = len(indices.shape) == 1
    if single_dim:
        indices = indices.unsqueeze(0)

    indices = indices.pad_fill(tokenizer.pad_token_id).tolist()
    text = tokenizer.batch_decode(indices, skip_special_tokens=True)
    text = [len(line.split()) for line in text]

    if single_dim:
        return text[0]
    else:
        return text


def _get_stats_of(values: List[int]) -> dict:
    return {
        'min': min(values),
        'mean': float(mean(values)),
        'stdev': float(std(values)),
        'max': max(values),
        'N': len(values)
    }


class Dataset:
    def __init__(self, path: str, langmodel: str = DEF_ENCODER, seed: int = 1, include_skip: bool = False):
        from transformers import AutoTokenizer
        import spacy

        # List of problem items
        self._whole_items: List[Example] = []
        # List of selected items
        self._items: List[Example] = []
        # Map from id to item
        self._id_map: Dict[str, Example] = {}
        # Map from experiments to set of ids (to manage splits)
        self._split_map: Dict[str, List[Example]] = {}
        # Vocab size of the tokenizer
        self._vocab_size: int = 0
        # Path of this dataset
        self._path = path
        # Lang Model applied in this dataset
        self._langmodel = langmodel
        self._langmodel_name = self._langmodel.replace('/','_')
        self.tokenizer = AutoTokenizer.from_pretrained(self._langmodel)
        self.nlp = spacy.load("en_core_web_sm")
        # RNG for shuffling
        self._rng = Generator(PCG64(seed))
        # include skip
        self.include_skip = include_skip

        # Read the dataset.
        cache_loaded = self._try_read_cache()
        if not cache_loaded:
            self._whole_items = self._save_cache()

        # Build id mapping
        self._id_map = {item.info.item_id: item
                        for item in self._whole_items}

    def _save_cache(self) -> List[Example]:
        # Otherwise, compute preprocessed result and cache it in the disk
        from json import load
        from torch import save

        # Make lock file
        save(True, str(self.cache_lock_path))

        # First, read the JSON with lines file.
        self._vocab_size = len(self.tokenizer)
        with Path(self._path).open('r+t', encoding='UTF-8') as fp:
            items = [Example.from_dict(item, self.tokenizer, self.nlp)
                     for item in load(fp)]

        # Cache dataset and vocabulary.
        save({
            CACHE_TOKENIZER: self._langmodel,
            CACHE_ITEMS: items,
            CACHE_VOCAB_SZ: self._vocab_size
        }, str(self.cached_path))

        # Delete lock file
        self.cache_lock_path.unlink()
        return items

    def _try_read_cache(self):
        while self.cache_lock_path.exists():
            # Wait until lock file removed (sleep 0.1s)
            sleep(0.1)

        if self.cached_path.exists():
            # If cached version is available, load the dataset from it.
            from torch import load
            cache = load(self.cached_path)

            if self._langmodel == cache[CACHE_TOKENIZER]:
                self._whole_items = cache[CACHE_ITEMS]
                self._vocab_size = cache[CACHE_VOCAB_SZ]
                return True

        return False

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def cached_path(self) -> Path:
        return Path(f'{self._path}_{self._langmodel_name}.cached')

    @property
    def cache_lock_path(self) -> Path:
        return Path(f'{self._path}_{self._langmodel_name}.lock')

    @property
    def num_items(self) -> int:
        return len(self._items)

    @property
    def get_dataset_name(self) -> str:
        return Path(self._path).stem

    @property
    def statistics(self) -> Dict[str, float]:
        return self.get_statistics()

    def keys(self) -> Set[str]:
        return {item.info.item_id
                for item in self._items}

    def get_statistics(self, as_whole: bool = True) -> Dict[str, float]:
        item_list = self._whole_items if as_whole else self._items
        return {
            'items': len(item_list),
            'text.words': _get_stats_of([_word_counts(item.text.tokens, self.tokenizer)
                                         for item in item_list]),
            'text.numbers': _get_stats_of([item.text.numbers.indices.max().item() + 1
                                           for item in item_list]),
            'equation.operators': _get_stats_of([item.equation.sequence_lengths.item()
                                                 for item in item_list]),
            'equation.var': _get_stats_of([item.equation.operator.indices.eq(OPR_NEW_VAR_ID).sum().item()
                                           for item in item_list])
        }

    def get_rng_state(self):
        return self._rng.__getstate__()

    def set_rng_state(self, state):
        self._rng.__setstate__(state)

    def reset_seed(self, seed):
        self._rng = Generator(PCG64(seed))

    def select_items_in(self, ids: List[str]):
        # Filter items and build task groups
        self._items = [self._id_map[_id]
                       for _id in ids if _id in self._id_map]

    def select_items_with_file(self, path: str):
        if path not in self._split_map:
            with Path(path).open('rt') as fp:
                items = [line.strip() for line in fp.readlines()]
                self.select_items_in(items)
                self._split_map[path] = self._items
        else:
            self._items = self._split_map[path]

    def get_minibatches(self, batch_size: int = 8, for_testing: bool = False):
        chunk_step = batch_size * 10
        batches = []

        items = self._items.copy()
        if not for_testing:
            self._rng.shuffle(items)

        for begin in range(0, len(items), chunk_step):
            chunk = items[begin:begin + chunk_step]
            chunk = sorted(chunk, key=Example.get_item_size)
            for start in range(0, len(chunk), batch_size):
                batches.append(Example.build_batch(*chunk[start:start+batch_size]))

        if not for_testing:
            self._rng.shuffle(batches)

        return batches


    @property
    def items(self):
        return self._items

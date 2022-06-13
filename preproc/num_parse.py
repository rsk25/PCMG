from nltk.corpus.reader import Synset
from sympy import Number
from typing import List, Optional, Tuple, Union

from nltk import download as nltk_download
from nltk.corpus import wordnet

from common.pen.pattern import NUMBER_OR_FRACTION_PATTERN, PUNCTUATION_END_PATTERN, ORDINAL_PATTERN

nltk_download('wordnet')


def get_synonyms(word: str, pos=wordnet.NOUN) -> List[str]:
    synsets: List[Synset] = wordnet.synsets(word, pos=pos)
    if not synsets:
        return []

    return [s for s in synsets[0].lemma_names()]


MISSING_CASES = {
    'a': 1, 'an': 1,

    'once': 1, 'twice': 2, 'thrice': 3,
    'double': 2, 'triple': 3, 'quadruple': 4,
    'doubled': 2, 'tripled': 3, 'quadrupled': 4,

    'half': 0.5, 'quarter': 0.25,
    'halved': 0.5, 'quartered': 0.25,
}

CARDINALS = {}
ORDINALS = {}
for _i in range(1, 100):
    _cardinal = get_synonyms('%d' % _i, wordnet.ADJ)
    if _i % 10 == 1 and _i // 10 != 1:
        _ordinal = get_synonyms('%dst' % _i, wordnet.ADJ)
    elif _i % 10 == 2 and _i // 10 != 1:
        _ordinal = get_synonyms('%dnd' % _i, wordnet.ADJ)
    elif _i % 10 == 3 and _i // 10 != 1:
        _ordinal = get_synonyms('%drd' % _i, wordnet.ADJ)
    else:
        _ordinal = get_synonyms('%dth' % _i, wordnet.ADJ)

    if _cardinal:
        CARDINALS[_cardinal[0]] = _i
    if _ordinal:
        # Add singular/plural forms
        ORDINALS[_ordinal[0]] = _i
        ORDINALS[_ordinal[0] + 's'] = _i


UNIT_PLACE = {}
for _d in range(2, 10):
    _cardinal = get_synonyms('1' + ('0' * _d))
    if _cardinal and '_' not in _cardinal[0]:
        # Add singular/plural forms
        UNIT_PLACE[_cardinal[0]] = _d
        UNIT_PLACE[_cardinal[0] + 's'] = _d


def _is_simple_number_format(word: str) -> Union[float, int, None]:
    if NUMBER_OR_FRACTION_PATTERN.fullmatch(word):
        # If this represents a decimal or fractional number
        if '/' in word:
            return eval(word)
        return float(word) if '.' in word else int(word)

    if word in CARDINALS:
        # If this is an cardinal without decimals
        return CARDINALS[word]

    if ORDINAL_PATTERN.fullmatch(word):
        # If this is an ordinal with decimals, like 23rd
        return int(word[:-2])

    if word in ORDINALS:
        # If this is an ordinal without decimals
        return ORDINALS[word]

    return None


def is_number(word: str, ignore_dash=False) -> Union[float, int, None]:
    word = word.lower()

    if word in MISSING_CASES:
        # 'first' is not linked with 1st in the wordnet.
        return MISSING_CASES[word]

    # Otherwise, find entire word from the wordnet.
    synonyms = [word] + get_synonyms(word)
    for word in synonyms:
        simple_format = _is_simple_number_format(word)
        if simple_format is not None:
            return simple_format

        if '_' in word or len(word) == 1:
            # Ignore words with spacing or words with a single character, like 'C' (greek number)
            continue

        if not ignore_dash and '-' in word:
            # [something]-[something]. Usually [number]-[unit] or [number]-[ordinal] (fraction)
            head, tail = word.split('-', 1)

            # Check whether the second thing indicates a fraction
            head = is_number(head, ignore_dash=True)
            if type(head) is int:
                if head == 1 and tail in ORDINALS:
                    return head / ORDINALS[tail]
                elif head != 1 and tail[:-1] in ORDINALS and tail.endswith('s'):
                    return head / ORDINALS[tail[:-1]]
                elif tail in {'half', 'halves'}:
                    return head / 2
                elif is_number(tail, ignore_dash=True) is None:
                    # This is a unit
                    return head

    return None


def find_numbers(text: str):
    numbers = []
    tokens = text.split()
    for tid, token in enumerate(tokens):
        token_no_paren = token[1:] if token.startswith('(') else token
        matched = NUMBER_OR_FRACTION_PATTERN.match(token_no_paren)
        if matched is not None:
            value = matched.group(0).replace(',', '')
            if '/' in value:
                value = eval(value)
            else:
                value = float(value) if '.' in value else int(value)
        else:
            no_punct = PUNCTUATION_END_PATTERN.sub('', token_no_paren)
            value = is_number(no_punct)

            if numbers and numbers[-1]['tokenRange'][-1] == tid - 1:
                last_number = numbers[-1]
                if no_punct in UNIT_PLACE:
                    # Number should be multiplied (hundred, thousand, million, ...)
                    last_number['token'].append(token)
                    last_number['value'] = last_number['value'] * value
                    last_number['tokenRange'].append(tid)
                    value = None
                elif no_punct in ORDINALS and last_number['token'][-1] in CARDINALS:
                    # Number should be divided (fraction)
                    last_number['token'].append(token)
                    last_number['value'] = last_number['value'] / value
                    last_number['tokenRange'].append(tid)
                    value = None
                elif no_punct in CARDINALS and last_number['token'][-1] in UNIT_PLACE:
                    # Number should be added (followed numbers, like one hundred "sixty-two")
                    last_number['token'].append(token)
                    last_number['value'] = last_number['value'] + value
                    last_number['tokenRange'].append(tid)
                    value = None

        if value is not None:
            numbers.append({
                'key': 'N_%02d' % len(numbers),
                'token': [token],
                'value': value,
                'tokenRange': [tid]
            })

    return numbers
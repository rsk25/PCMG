import re
from typing import List, Union, Tuple

from common.const.operand import PREFIX_LEN
from common.const.pad import PAD_ID
from common.pen.pattern import NUMBER_OR_FRACTION_PATTERN as NOF_WITH_START_PATTERN
from transformers.file_utils import add_start_docstrings
from .base import *
from .label import Label

# Remove '^' from the pattern.
NUMBER_OR_FRACTION_PATTERN = re.compile(NOF_WITH_START_PATTERN.pattern[1:])


def _number_index_reader(token: int) -> str:
    if token == PAD_ID:
        return '__'
    else:
        return '%2d' % token


def _add_space_around_number(text: str) -> Tuple[str, dict]:
    orig_to_spaced = {}
    spaced = []
    for orig, token in enumerate(text.split()):
        spaced_tokens = re.sub('\\s+', ' ', NUMBER_OR_FRACTION_PATTERN.sub(' \\1 ', token)).strip().split()
        orig_to_spaced[orig] = len(spaced)
        token_reset = False

        for new in spaced_tokens:
            if not token_reset and NUMBER_OR_FRACTION_PATTERN.fullmatch(new):
                orig_to_spaced[orig] = len(spaced)
                token_reset = True
            # Add token
            spaced.append(new)

    return ' '.join(spaced), orig_to_spaced


def _remove_special_prefix(token: str) -> str:
    # Handle different kind of spacing prefixes...
    from transformers import SPIECE_UNDERLINE
    if token == SPIECE_UNDERLINE:
        return ' '
    if token.startswith(SPIECE_UNDERLINE):
        return token[len(SPIECE_UNDERLINE):]
    if token.startswith('##'):
        return token[2:]
    return token

def text_tokenization(string: str, tokenizer) -> Tuple[str, dict, List[int]]:
    spaced, orig_to_new_wid = _add_space_around_number(string)
    tokens: List[int] = tokenizer.encode(spaced)
    return spaced, orig_to_new_wid, tokens


def gather_number_toks(tokens: List[int], spaced_text: str, orig_to_new_wid: dict, \
                       numbers: dict, tokenizer) -> List[int]:
    # Read numbers
    wid_to_nid = {orig_to_new_wid[token_id]: int(number['key'][PREFIX_LEN:])
                    for number in numbers
                    for token_id in number['tokenRange']}

    # Find position of numbers
    token_nids = []
    current_nid = PAD_ID
    current_wid = 0
    string_left = ' ' + spaced_text.lower()
    for token in tokenizer.convert_ids_to_tokens(tokens):
        if token in tokenizer.all_special_tokens:
            current_nid = PAD_ID
        else:
            # Find whether this is the beginning of the word.
            # We don't use SPIECE_UNDERLINE or ## because ELECTRA separates comma or decimal point...
            if string_left[0].isspace():
                current_nid = wid_to_nid.get(current_wid, PAD_ID)
                current_wid += 1
                string_left = string_left[1:]

            token_string = _remove_special_prefix(token)
            assert string_left.startswith(token_string)
            string_left = string_left[len(token_string):]

        token_nids.append(current_nid)

    return token_nids


def gather_text_toks(tokens, tokenizer) -> List[int]:
    # Set pad token id as PAD_ID (This will be replaced inside a model instance)
    tokens = [tok if tok != tokenizer.pad_token_id else PAD_ID
                for tok in tokens]
    return tokens


class Text(TypeTensorBatchable, TypeSelectable):
    #: Tokenized raw text (tokens are separated by whitespaces)
    raw: Union[str, List[str]]
    #: Tokenized text
    tokens: Label
    #: Number index label for each token
    numbers: Label
    #: Number keywords for each number label
    keywords: Union[Label, List[Label]]
    #: Equation(s) that will be used as prompt
    prompt_eq: Label

    def __init__(self, raw: Union[str, List[str]], tokens: Label, numbers: Label, keywords: Union[Label, List[Label]], prompt_eq: Label):
        super().__init__()
        self.raw = raw
        self.tokens = tokens
        self.numbers = numbers
        self.keywords = keywords
        self.prompt_eq = prompt_eq

    def __getitem__(self, item) -> 'Text':
        if type(item) is int and self.is_batched:
            return Text(raw=self.raw[item], tokens=self.tokens[item], numbers=self.numbers[item],
                        keywords=self.keywords[item], prompt_eq=self.prompt_eq[item])
        else:
            return super().__getitem__(item)

    @property
    def shape(self) -> torch.Size:
        return self.tokens.shape

    @property
    def pad(self) -> torch.BoolTensor:
        return self.tokens.pad

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    @property
    def sequence_lengths(self) -> torch.LongTensor:
        return self.tokens.sequence_lengths

    @classmethod
    def build_batch(cls, *items: 'Text') -> 'Text':
        return Text(raw=[item.raw for item in items],
                    tokens=Label.build_batch(*[item.tokens for item in items]),
                    numbers=Label.build_batch(*[item.numbers for item in items]),
                    keywords=Label.build_batch(*[item.keywords for item in items]),
                    prompt_eq=Label.build_batch(*[item.prompt_eq for item in items]))

    @classmethod
    def from_dict(cls, raw: dict, tokenizer, nlp) -> 'Text':
        # Tokenize the text
        spaced, orig_to_new_wid, tokens = text_tokenization(raw['text'], tokenizer)
        text: str = ' '.join(tokenizer.convert_ids_to_tokens(tokens, skip_special_tokens=True))
        token_nids = gather_number_toks(tokens, spaced, orig_to_new_wid, \
                                        raw['numbers'], tokenizer)
        tokens = gather_text_toks(tokens, tokenizer)
        assert len(tokens) == len(token_nids)

        # Extract keywords: keywords will be save as raw string because it will later be tokenized
        # during the special embedding process
        keywords = []
        exclude_words = ['how', 'if', 'when', 'what', 'he', 'she']
        all_stopwords = nlp.Defaults.stop_words
        text_lower_split = text.lower().split()
        for tok in text_lower_split:
            # remove punctuation and stopwords
            if tok.isalpha() and (len(tok) > 1) and (tok not in all_stopwords) and (tok not in exclude_words): 
                keywords.append(tok)
        if len(keywords) == 0:
            for tok in text_lower_split:
                if tok.isalpha() and (len(tok) > 1):
                    keywords.append(tok)
        assert len(keywords) != 0
        keywords_text = ' '.join(set(keywords))
        keywords_enc = tokenizer.encode(keywords_text, add_special_tokens=False)

        # Extract Equations for text prompt
        if type(raw['oldFormula']) is list and len(raw['oldFormula']) > 1:
            _eq = ', '.join(raw['oldFormula'])
        elif type(raw['oldFormula']) is str:
            _eq = raw['oldFormula']
        else:
            _eq = raw['oldFormula'][0]
        prompt_eq = tokenizer.encode(_eq, add_special_tokens=False)
        
        return Text(raw=text, tokens=Label.from_list(tokens), numbers=Label.from_list(token_nids),
                    keywords=Label.from_list(keywords_enc), prompt_eq=Label.from_list(prompt_eq))

    def as_dict(self) -> dict:
        return dict(raw=self.raw, tokens=self.tokens, numbers=self.numbers, keywords=self.keywords, prompt_eq=self.prompt_eq)

    def to_human_readable(self, tokenizer=None) -> dict:
        if tokenizer is None:
            text_converter = None
        else:
            text_converter = lambda t: tokenizer.convert_ids_to_tokens(t) if t != PAD_ID else ''

        return {
            'raw': human_readable_form(self.raw),
            'tokens': self.tokens.to_human_readable(converter=text_converter),
            'numbers': self.numbers.to_human_readable(converter=_number_index_reader),
            'keywords': human_readable_form(self.keywords, converter=text_converter),
            'prompt_eq': human_readable_form(self.prompt_eq, converter=text_converter)
        }

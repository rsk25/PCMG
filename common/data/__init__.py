from .base import *
from .explanation import *
from .encoded import *
from .equation import *
from .label import *
from .prediction import *
from .text import *


def _compute_accuracy_from_list(items: list, key: str = '') -> dict:
    values = {}
    for tgt in {'token', 'seq'}:
        for field in {'corrects', 'total'}:
            values[tgt + field] = items[tgt][field]

    return {
        f'token_acc_{key}': values['tokencorrects'] / values['tokentotal'] if values['tokentotal'] else float('NaN'),
        f'seq_acc_{key}': values['seqcorrects'] / values['seqtotal'] if values['seqtotal'] else float('NaN')
    }


class ExtraInfo(TypeBase):
    item_id: str
    answers: List[Dict[str, sympy.Number]]
    numbers: Dict[str, sympy.Number]
    variables: List[str]
    split: Optional[str]
    raw: Optional[dict]

    def __init__(self, item_id: str, answers: List[Dict[str, sympy.Number]], numbers: Dict[str, sympy.Number],
                 variables: List[str], split: str = None, raw: dict = None):
        super().__init__()
        self.item_id = item_id
        self.answers = answers
        self.numbers = numbers
        self.variables = variables
        self.split = split
        self.raw = raw

    @classmethod
    def from_dict(cls, raw: dict) -> 'ExtraInfo':
        answers = [{key: sympy.Number(var)
                    for key, var in item.items() if not key.startswith('_')}
                   for item in raw['answers'] if item['_selected']]
        numbers = {num['key']: sympy.Number(num['value'])
                   for num in raw['numbers']}
        return ExtraInfo(item_id=raw['_id'], split=None, answers=answers, numbers=numbers, variables=[], raw=raw)

    def filter_answers(self) -> 'ExtraInfo':
        kwargs = self.as_dict()
        kwargs['answers'] = [{key: value
                              for key, value in answer_tuple.items()
                              if key in kwargs['variables']}
                             for answer_tuple in kwargs['answers']]
        return ExtraInfo(**kwargs)

    def as_dict(self) -> dict:
        return dict(item_id=self.item_id, answers=self.answers, numbers=self.numbers, variables=self.variables,
                    split=self.split, raw=self.raw)

    def to_human_readable(self) -> dict:
        result = self.as_dict()
        result.pop('raw')
        return human_readable_form(result)


class Example(TypeBatchable):
    text: Text
    equation: Equation
    explanation: Union[Explanation, List[Explanation]]
    info: Union[ExtraInfo, List[ExtraInfo]]

    def __init__(self, text: Text, equation: Equation, info: Union[ExtraInfo, List[ExtraInfo]]):
        super().__init__()
        self.text = text
        self.equation = equation
        self.info = info

    @property
    def device(self) -> torch.device:
        return self.text.device

    @property
    def is_batched(self) -> bool:
        return self.text.is_batched

    @property
    def batch_size(self):
        return self.text.shape[0] if self.is_batched else 1

    @classmethod
    def build_batch(cls, *items: 'Example') -> 'Example':
        return Example(text=Text.build_batch(*[item.text for item in items]),
                       equation=Equation.build_batch(*[item.equation for item in items]),
                       info=[item.info for item in items])

    @classmethod
    def concat(cls, *items: 'Example') -> 'Example':
        raise NotImplementedError('This operation is not supported')

    @classmethod
    def from_dict(cls, raw: dict, tokenizer, nlp) -> 'Example':
        _info = ExtraInfo.from_dict(raw)
        _text = Text.from_dict(raw, tokenizer=tokenizer, nlp=nlp)
        _equation = Equation.from_dict(raw, var_list_out=_info.variables)

        # Filter out not-used variables from the answers
        _info = _info.filter_answers()

        return Example(text=_text, equation=_equation, info=_info)

    def as_dict(self) -> dict:
        return dict(text=self.text, equation=self.equation, info=self.info)

    def get_item_size(self) -> int:
        return max(self.text.shape[-1], self.equation.shape[-1])

    def item_of_batch(self, index: int) -> 'Example':
        assert self.is_batched
        return Example(text=self.text[index],
                       equation=self.equation[index],
                       info=self.info[index])

    def to_human_readable(self, tokenizer=None) -> dict:
        if self.is_batched:
            return dict(
                info=[i.to_human_readable() for i in self.info],
                text=self.text.to_human_readable(tokenizer),
                equation=self.equation.to_human_readable(),
            )
        else:
            return dict(
                info=self.info.to_human_readable(),
                text=self.text.to_human_readable(tokenizer),
                equation=self.equation.to_human_readable(),
            )

    def accuracy_of(self, **kwargs) -> dict: ### Fix here (rsk25)
        # equation: EquationPrediction [B, T]
        # 
        result = {}
        if 'equation' in kwargs:
            if 'equation_tgt' in kwargs:
                eqn_tgt = kwargs.pop('equation_tgt')
            elif 'eqn_ignore' in kwargs:
                eqn_tgt = self.equation.ignore_labels(kwargs.pop('eqn_ignore'))
            else:
                eqn_tgt = self.equation
            result.update(eqn_tgt.accuracy_of(kwargs.pop('equation')))

        # if 'mwp' in kwargs:
        #     mwp_cnt = self.text.tokens.num_corrects(kwargs['mwp'])
        #     result.update(_compute_accuracy_from_list(mwp_cnt, key='mwp_generated'))

        return result

    def loss_calculation(self, kw_kl_prior: float, kw_kl_coef: float, **kwargs) -> Dict[str, torch.Tensor]: ### Fix here (rsk25)
        # equation: EquationPrediction [B, T]
        # 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = {}
        if 'equation' in kwargs:
            if 'equation_tgt' in kwargs:
                eqn_tgt = kwargs.pop('equation_tgt')
            elif 'eqn_ignore' in kwargs:
                eqn_tgt = self.equation.ignore_labels(kwargs.pop('eqn_ignore'))
            else:
                eqn_tgt = self.equation
            result.update(eqn_tgt.smoothed_cross_entropy(kwargs.pop('equation'), smoothing=0.01))

        if 'kw_logits' in kwargs:
            ### The reason we use KL Divergence is because we do not have a dataset this model can use to learn.
            ### Instead, we have set a prior distribution (Bernoulli) so we can use that to calculate the loss. 
            p = torch.sigmoid(kwargs.pop('kw_logits'))[:,0]
            kw_kl_loss = p * torch.log( p / torch.tensor([kw_kl_prior]).to(device)) \
                + (1-p) * torch.log( (1-p) / torch.tensor([1-kw_kl_prior]).to(device))
            kw_kl_loss = kw_kl_loss.mean()

            result['keyword_loss'] = kw_kl_coef * kw_kl_loss

        return result


__all__ = ['Example', 'Text', 'Equation', 'EquationPrediction',
           'ExtraInfo', 'Encoded', 'Label', 'Prediction']

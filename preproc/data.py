from typing import List, Dict, Union
import re

from common.const.operand import NUM_FORMAT

NUM_FORMAT = 'N_'
NUMBERS_DEFAULT_FORMAT = {
                "key": '',
                "token": [],
                "tokenRange": [],
                "value": ''
            }

def to_pen_decimal(fraction_str: str) -> float:
    pass

class MathWordProblem:
    def __init__(self, oldText: str, oldFormula: List[str], oldAnswer: List[str]) -> None:
        # Un-determined: use pre-determined data to determine
        self._id: str = None
        self.text: str = None
        self.equations: List[str] = None
        self.numbers: Dict[str, Union[List[str], str]] = []
        self.answers: List[Dict[str, Union[str,bool]]] = [{"_selected": True}]
        self.index: int = None
        self.dataset: str = None
        # Pre-determined
        self.oldText = oldText
        self.oldFormula = oldFormula
        self.oldAnswer = oldAnswer
        self.lang ="en"
        self._exclude = False
        self.worker_for_train = ""
        self.explanations = {}

    def set_text(self):
        raise NotImplementedError()

    def set_answer(self):
        raise NotImplementedError()

    def set_numbers(self):
        raise NotImplementedError()

    def set_equation(self):
        raise NotImplementedError()

    def as_dict(self):
        raise NotImplementedError()


class MathWordDataset:
    def __init__(self):
        pass

    def set_ids(self):
        raise NotImplementedError()

    def set_indices(self):
        raise NotImplementedError()


class Math23kProblem(MathWordProblem):
    def __init__(self, oldText: str, oldFormula: List[str], oldAnswer: List[str], mwp_template: str, eqs_template: List[str]) -> None:
        super().__init__(oldText, oldFormula, oldAnswer)
        self.dataset = "math23k"

        # attributes that are used as templates for parsing
        self.mwp_template = mwp_template
        self.eqs_template = eqs_template
        self.is_set = False

    def _math23k_to_pen_format(math23k_str: str) -> str:
        pen_pattern1 = r'num(\d{2})'
        pen_pattern2 = r'num(\d{1})'
        math23k_str = re.sub(pen_pattern1, NUM_FORMAT+r'\1', math23k_str)
        pen_str = re.sub(pen_pattern2, NUM_FORMAT+r'0\1', math23k_str)
        return pen_str

    def set_text(self) -> None:
        self.text = self.oldText

    def set_answer(self) -> None:
        self.answers[0].update({"x": int(self.oldAnswer[0])})

    def set_numbers(self) -> None:
        # TODO: use mwp_template and eps_template
        # key, token, tokenRange
        pair_for_parsing = zip(self.oldText.split(), self.mwp_template.split())
        p1 = r'^num(\d{1,2})$'
        
        for i, orig_text, labeled_text in enumerate(pair_for_parsing):
            if re.match(p1, labeled_text):
                self.numbers.append(NUMBERS_DEFAULT_FORMAT)
                self.numbers[-1].update({'key': self._math23k_to_pen_format(labeled_text)})
                self.numbers[-1].update({'token': [orig_text]})
                self.numbers[-1].update({'tokenRange':[i]})
                self.numbers[-1].update({'value': to_pen_decimal(orig_text)})
        
    def set_equations(self) -> None:
        eqs = self.eqs_template
        new_eqs = [self._math23k_to_pen_format(eq) for eq in eqs]
        self.equations = new_eqs

    def set_exclude(self) -> None:
        exclude_pattern = "="
        if exclude_pattern in self.text:
            self._exclude = True

    def set_all(self) -> None:
        self.set_text()
        self.set_answer()
        self.set_numbers()
        self.set_equations()
        self.set_exclude()
        self.is_set = True

    def as_dict(self) -> dict:
        if self.is_set is False:
            self.set_all()

        return dict(
            _exclude=self._exclude,
            _id=self._id,
            answer=self.answers,
            dataset=self.dataset,
            equations=self.equations,
            index=self.index,
            lang=self.lang,
            numbers=self.numbers,
            oldAnswer=self.oldAnswer,
            oldFormula=self.oldFormula,
            oldText=self.oldText,
            text=self.text,
            worker_for_train=self.worker_for_train,
            explanations=self.explanations
        )
    
class Math23kDataset(MathWordDataset):
    def __init__(self):
        super().__init__()
        pass

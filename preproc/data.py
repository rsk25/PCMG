import json
import pickle
from typing import List, Dict, Union
import re
from copy import deepcopy

from sympy.core import sympify
from sympy.core.numbers import Integer, Float

from common.const.operand import NUM_PREFIX
from common.pen.pattern import NUMBER_OR_FRACTION_PATTERN
from sympy.solvers import ode


NUMBERS_DEFAULT_FORMAT = {
                "key": '',
                "token": [],
                "tokenRange": [],
                "value": ''
            }



def to_pen_decimal(number_str: str) -> float:
    assert NUMBER_OR_FRACTION_PATTERN.fullmatch(number_str)
    number_str = re.sub(',','',number_str)
    number = sympify(number_str, evaluate=True, rational=False)
    return number


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


    def _math23k_to_pen_format(self, math23k_str: str) -> str:
        pattern1 = r'num(\d{2})'
        pattern2 = r'num(\d{1})'
        pen_str = re.sub(pattern1, NUM_PREFIX+r'\1', math23k_str)
        pen_str = re.sub(pattern2, NUM_PREFIX+r'0\1', pen_str)
        return pen_str

    
    def set_text(self) -> None:
        self.text = self.oldText


    def set_answer(self) -> None:
        self.answers[0].update({"x": int(self.oldAnswer[0])})


    def set_equations(self) -> None:
        assert self.numbers is not [], "Numbers need to be set!"
        assert self.oldFormula is not None, "oldFormula needs to be set!"

        # collect keys
        key_stack = []
        for number in self.numbers:
            key_stack.append(number.get('key'))
        assert len(key_stack) != 0

        new_eqs = []
        op_pattern = r"(\d+%{1}(?=[-*/+={}()%])|\d+%{1}$|[a-zA-Z-*/+={}()%+]|\d+(?!%))"
        num_pattern = r'num\d{1,2}'
        for old_formula, eq_temp in zip(self.oldFormula, self.eqs_template):
            oldFormula_spaced = re.sub(op_pattern, r"\1 ", old_formula).rstrip()
            oldFormula_split = oldFormula_spaced.split()
            eqs_template_split = eq_temp.split()
            assert len(oldFormula_split) == len(eqs_template_split)
            
            _eq = []
            for orig_op, _op in zip(oldFormula_split, eqs_template_split):
                if re.fullmatch(num_pattern, _op):
                    new_op = self._math23k_to_pen_format(_op)
                    if new_op in key_stack:
                        _eq.append(new_op)
                    else:
                        _eq.append(orig_op)
                else:
                    _eq.append(orig_op)

            new_eqs.append(' '.join(_eq))

        self.equations = new_eqs


    def set_numbers(self) -> None:

        def _update_numbers(_number ,key_name: str, update_value: Union[List[Union[str, int]], str]):
            assert type(update_value) in [list, str, Integer, Float], f"update_value is {type(update_value)}"
            _number.update({key_name: update_value})

        def delete_whitespace(template: str) -> str:
            template = re.sub(r"(\S+)\s+([-.,!?')};:]+)", r"\1\2", template)
            template = re.sub(r"([-.,!?')};:]+)\s+([-.,!?')};:]+)", r"\1\2", template)
            template = re.sub(r"([-({']+)\s+(\S+)", r"\1\2", template)
            return template
        
        self.mwp_template = delete_whitespace(self.mwp_template)
        mwp_template_split = self.mwp_template.split()
        oldText_split = self.oldText.split()
        assert len(mwp_template_split) == len(oldText_split), \
            f"MWP template does not match oldText by {abs(len(mwp_template_split) - len(oldText_split))} tokens!\n"
        pair_for_parsing = zip(oldText_split, mwp_template_split)
        p1 = r'num(\d{1,2})'

        for i, (orig_text, labeled_text) in enumerate(pair_for_parsing):
            if re.fullmatch(p1, labeled_text):
                new_number = deepcopy(NUMBERS_DEFAULT_FORMAT)
                _update_numbers(new_number, 'key', self._math23k_to_pen_format(labeled_text))
                _update_numbers(new_number, 'token', [orig_text])
                _update_numbers(new_number, 'tokenRange', [i])
                _update_numbers(new_number, 'value', to_pen_decimal(orig_text))
                self.numbers.append(new_number)
                

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

    
    def as_dict(self):
        pass


    def to_json(self, json_name: str) -> None:
        with open(json_name, 'w+', encoding='utf-8') as json_writer:
            json.dump(json_writer, self.as_dict(), ensure_ascii=False)


    def to_pickle(self, pkl_name: str) -> None:
        with open(pkl_name, 'wb+', encoding='utf-8') as pkl_writer:
            pickle.dump(pkl_writer, self.as_dict())

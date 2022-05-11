import json
import pickle
from typing import List, Dict, Union
import re
from copy import deepcopy

from sympy.core import sympify
from sympy.core.numbers import Integer, Float

from common.const.operand import NUM_PREFIX
from common.pen.pattern import NUMBER_OR_FRACTION_PATTERN


NUMBERS_DEFAULT_FORMAT = {
                "key": '',
                "token": [],
                "tokenRange": [],
                "value": ''
            }

WORD_NUMBERS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'twelve': 12,
    'thirteen': 13,
    'sixteen': 16,
    'nineteen': 19,
    'fifty': 50,
    'hundred': 100,
    'thousand': 1000,
    'thirty': 30,
    'forty': 40,
    'twenty': 20
}

def to_pen_decimal(number_str: str) -> float:
    number_str = re.sub(r"\(|\)|°C|/min|/hour", "", number_str)
    number_str = re.sub(r"^\(|[.,?!);\"]$", "", number_str)
    number_str = re.sub(',', '', number_str)
    number_str = re.sub(r"(\d+(\.\d+)?)%", r"\1*100", number_str)
    try:
        number = eval(number_str)
    except:
        number = WORD_NUMBERS.get(number_str.lower())
        assert number != None
    return number


class MathWordProblem:
    def __init__(self, oldText: str, oldFormula: List[str], oldAnswer: List[str]) -> None:
        # Un-determined: use pre-determined data to determine
        self._id: str = None
        self._represent: bool = False
        self.text: str = None
        self.equations: List[str] = []
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
        self.start_id: int = None
        self.start_index: int = None
        self.problems = []

    @property
    def number_of_problems(self):
        return len(self.problems)


    def append_to_dataset(self, problems: 'MathWordProblem') -> None:
        self.problems.append(problems.as_dict())


class Math23kProblem(MathWordProblem):
    def __init__(self, problem_id: str, oldText: str, oldFormula: List[str], oldAnswer: List[str], mwp_template: str, eqs_template: List[str]) -> None:
        super().__init__(oldText, oldFormula, oldAnswer)
        self.dataset = "math23k"
        self.problem_id = int(problem_id)

        # attributes that are used as templates for parsing
        self.orig_mwp_template = mwp_template
        self.orig_eqs_template = eqs_template
        self.mwp_template = mwp_template
        self.eqs_template = eqs_template
        self.is_set = False


    def _math23k_to_pen_format(self, math23k_str: str) -> str:
        pattern1 = r'num(\d{2})([%(),?]|\.$)?'
        pattern2 = r'num(\d{1})([%(),?]|\.$)?'
        pen_str = re.sub(pattern1, NUM_PREFIX+r'\1', math23k_str)
        pen_str = re.sub(pattern2, NUM_PREFIX+r'0\1', pen_str)
        pen_str = re.sub(r".*(N_0\d{1}|N_\d{2}).*",r"\1", pen_str)
        return pen_str

    
    def set_text(self) -> None:
        _text = re.sub(r"\((\d+/\d+)\)",r"\1", self.oldText)
        _text = re.sub(r"\"", "", _text)
        _text = re.sub(r"(\d+)([a-zA-Z])", r"\1 \2", _text)
        _text = re.sub(r"(\w)-(\w)", r"\1 - \2", _text)
        _text = re.sub(r"(\w):([\w\"\s])", r"\1 : \2", _text)
        _text = re.sub(r"(\w):([\w\"\s])", r"\1 : \2", _text)
        self.text = _text

    def set_answer(self) -> None:
        temp_answer = to_pen_decimal(self.oldAnswer[0])
        if temp_answer - int(temp_answer) == 0.0:
            self.answers[0].update({"x": int(temp_answer)})
        else:
            self.answers[0].update({"x": float(temp_answer)})


    def set_equations(self) -> None:
        if self._exclude:
            return None
        assert len(self.numbers) != 0, "Numbers need to be set!"
        assert self.oldFormula is not None, "oldFormula needs to be set!"

        # collect keys, tokens and values
        key_stack = []
        key_token_range_value_stack = []
        for number in self.numbers:
            key_stack.append(number.get('key'))
            key_token_range_value_stack.append(
                (
                    number['key'], 
                    number['token'], 
                    number['tokenRange'], 
                    number['value']
                )
            )
        
        assert len(key_token_range_value_stack) != 0 and len(key_stack) != 0
        
        new_eqs = []
        op_pattern = r"(\(\d+/\d+\)|\d+\.\d+%{1}|\d+\.\d+|\d+%{1}(?=[-*/+={}\[\]()%])|\d+%{1}$|\d+(?!%)|\d+|[a-zA-Z-*/+={}\[\]()%+])"
        num_pattern = r'num\d{1,2}'
        for old_formula, eq_temp in zip(self.oldFormula, self.eqs_template):
            oldFormula_spaced = re.sub(op_pattern, r"\1 ", old_formula).rstrip()
            oldFormula_spaced = re.sub(r"\[", r"(", oldFormula_spaced)
            oldFormula_spaced = re.sub(r"\]", r")", oldFormula_spaced)
            oldFormula_split = oldFormula_spaced.split()

            new_oldFormula_split = []
            for op in oldFormula_split:
                if re.match(r"\(\d+/\d+\)", op) and (op[1:-1] not in self.oldText):
                    new_ops = re.sub(r"\((\d+)/(\d+)\)", r"( \1 / \2 )", op)
                    new_oldFormula_split += new_ops.split()
                else:
                    new_oldFormula_split.append(op)

            eqs_template_split = eq_temp.split()

            assert len(new_oldFormula_split) == len(eqs_template_split), \
                f"oldFormula: {new_oldFormula_split},\n \
                eqs_template: {eqs_template_split}"
            
            _eq = []
            for orig_op, _op in zip(new_oldFormula_split, eqs_template_split):
                if re.fullmatch(num_pattern, _op):
                    new_op = self._math23k_to_pen_format(_op)
                    if new_op in key_stack:
                        key_token_range_value_tuple = [t for t in key_token_range_value_stack if t[0] == new_op][0]
                        tok_range = key_token_range_value_tuple[2][0]
                        text_list = self.text.split()
                        if '%' in text_list[tok_range]:
                            _eq.append(new_op+" * 0.01")
                        else:
                            _eq.append(new_op)
                    else:
                        _eq.append(orig_op)
                else:
                    _eq.append(orig_op)

            new_eqs.append(' '.join(_eq))

        for eqs in new_eqs:
            if re.search(r" [03-9]+ | [12][^\s]| [1-24-9]+\.\d+| 0\.[^01]|\.\.\.|\d{3}|(?<![N_.])\d{2}|\d+/\d+", eqs):
                self._exclude = True
            else:
                self.equations.append(eqs)


    def set_numbers(self) -> None:

        def delete_whitespace(template: str) -> str:
            template = re.sub(r"\"", "", template)
            template = re.sub(r"(\w+)\s+([.,!?')};])", r"\1\2", template)
            template = re.sub(r"([.,!?')};]+)\s+([.,!?')};])", r"\1\2", template)
            template = re.sub(r"([({']+)\s+(\w+)", r"\1\2", template)
            template = re.sub(r"(?<!\d\s)(-) (num\d{1,2})", r"\1\2", template)
            template = re.sub(r"(s')(\S+)", r"\1 \2", template)
            return template
        
        self.mwp_template = delete_whitespace(self.mwp_template)
        mwp_template_split = self.mwp_template.split()
        oldText_split = self.text.split()
        assert len(mwp_template_split) == len(oldText_split), \
            f"MWP template does not match oldText by {abs(len(mwp_template_split) - len(oldText_split))} tokens!\n"
        pair_for_parsing = zip(oldText_split, mwp_template_split)
        p1 = r'num(\d{1,2})'

        for i, (orig_text, labeled_text) in enumerate(pair_for_parsing):
            if re.search(p1, labeled_text):
                new_number = deepcopy(NUMBERS_DEFAULT_FORMAT)
                new_number.update({'key': self._math23k_to_pen_format(labeled_text)})
                new_number.update({'token': [re.sub(r"[%(),?:]|\.$", "",orig_text)]})
                new_number.update({'tokenRange': [i]})
                new_number.update({'value': to_pen_decimal(orig_text)})
                self.numbers.append(new_number)
        
        if len(self.numbers) == 0:
            self._exclude = True
            self.answers = [{"_selected": True}]
            self.numbers = []
            self.equations = None
                

    def set_exclude(self) -> None:
        exclude_text_pattern = r"[=*-+]|\(\d+\)|\(\)|[△○□]|\.\.\."
        exclude_eq_pattern = r"[^=*\-+/%()\[\]{}.\w]"
        keywords = r"what|when|how|many|much|\?"
        if re.search(exclude_text_pattern, self.text):
            self._exclude = True
        if re.search(keywords, self.text) is None:
            self._exclude = True
        if re.search(exclude_eq_pattern, self.oldFormula[0]):
            self._exclude = True



    def set_all(self) -> None:
        self.set_text()
        self.set_exclude()
        if not self._exclude:
            self.set_answer()
            self.set_numbers()
            self.set_equations()
        self.is_set = True


    def as_dict(self) -> dict:
        if self.is_set == False:
            self.set_all()

        return dict(
            _exclude=self._exclude,
            _id=self._id,
            _represent=self._represent,
            answers=self.answers,
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
            explanations=self.explanations,
        )
    

class Math23kDataset(MathWordDataset):
    def __init__(self):
        super().__init__()
        self.start_id = int("60f9186c90be09b72239f2a5", base=16) + 1
        self.start_index = None
    

    def append_to_dataset(self, problem: 'Math23kProblem') -> None:
        problem._id = hex(self.start_id + self.number_of_problems)[2:]
        self.problems.append(problem.as_dict())

    
    def stack(self, problem: 'Math23kProblem') -> None:
        if not problem._exclude:
            problem._id = hex(self.start_id + self.number_of_problems)[2:]
            problem.set_all()
            self.problems.append(problem)


    def to_json(self, json_name: str) -> None:
        with open(json_name, 'w+', encoding='utf-8') as json_writer:
            json.dump(json_writer, self.problems, ensure_ascii=False)


    def to_pickle(self, pkl_name: str) -> None:
        with open(pkl_name, 'wb+', encoding='utf-8') as pkl_writer:
            pickle.dump(pkl_writer, self.problems)

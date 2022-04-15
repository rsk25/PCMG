from typing import List, Dict, Union


class MathWordProblem:
    def __init__(self, oldText: str, oldFormula: List[str], oldAnswer: List[str]) -> None:
        # Un-determined: use pre-determined data to determine
        self._id: str = None
        self.text: str = None
        self.equations: List[str] = None
        self.numbers: Dict[str, Union[List[str], str]] = {
            "key": '',
            "token": [],
            "tokenRange": [],
            "value": ''
        }
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

    def set_id(self):
        raise NotImplementedError()

    def set_index(self):
        raise NotImplementedError()

    def set_answer(self):
        raise NotImplementedError()

    def set_numbers(self):
        raise NotImplementedError()

    def as_dict(self):
        raise NotImplementedError()


class Math23kProblem(MathWordProblem):
    def __init__(self, oldText: str, oldFormula: List[str], oldAnswer: List[str], mwp_template: str, eqs_template: List[str]) -> None:
        super().__init__(oldText, oldFormula, oldAnswer)
        self.dataset = "math23k"

        # attributes that are used as templates for parsing
        self.mwp_template = mwp_template
        self.eqs_template = eqs_template
        self.is_set = False

    def set_text(self):
        self.text = self.oldText

    def set_id(self):
        # TODO: random number generator
        self._id = 0

    def set_index(self):
        # TODO: continue from the last index of PEN?
        self.index = 0

    def set_answer(self):
        self.answers[0].update({"x": int(self.oldAnswer[0])})

    def set_numbers(self):
        # TODO: use mwp_template and eps_template
        self.numbers = self.numbers

    def set_all(self):
        self.set_text()
        self.set_id()
        self.set_index()
        self.set_answer()
        self.set_numbers()
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
            oldFormula=self.oldForumla,
            oldText=self.oldText,
            text=self.text,
            worker_for_train=self.worker_for_train,
            explanations=self.explanations
        )
    
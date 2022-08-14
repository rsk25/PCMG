from collections import defaultdict
from logging import Logger
from typing import List, Tuple

import sympy

from .const.operator import OPR_NEW_VAR_ID
from .data import Example
from .pen.metric import compute_metrics
from .pen.solve import Solver


class Tester:
    def __init__(self, error_limit: float = 1E-3, time_limit: float = 5, logger: Logger = None):
        super().__init__()
        # AnswerChecker instance to verify the answer
        self._checker = Solver(error_limit, time_limit, logger)

    def close(self):
        self._checker.close()

    def check(self, output_pairs: List[Tuple[Example, dict]], tokenizer=None) -> dict:
        eqn_metrics = defaultdict(list)
        result_dumps = []

        mwp_references = {}
        mwp_hypotheses = {}

        for item, pair in output_pairs:
            result_dump_b = {}
            if tokenizer is not None:
                result_dump_b = item.to_human_readable(tokenizer)

            # (1) Collect generated MWPs if possible.
            expected, predicted = pair['mwp']
            result_dump_b['mwp_generated'] = predicted
            id_prefix = item.info.item_id

            mwp_references.update({
                id_prefix : expected
            })

            mwp_hypotheses.update({
                id_prefix : predicted
            })

            # (2) compute accuracy of the equation
            if 'correctness' in pair:
                correct = pair['correctness']
            else:
                expected, predicted = pair['equation']
                result_dump_b['eqn_generated'] = predicted.to_human_readable()

                # (2-1) Count the number of variables
                var_count_pred = predicted.operator.indices.eq(OPR_NEW_VAR_ID).long().sum().item()
                var_count_orig = expected.operator.indices.eq(OPR_NEW_VAR_ID).long().sum().item()
                eqn_metrics['var'].append(var_count_pred - var_count_orig)

                # (2-2) Check the answer
                result, exception = self._checker.solve(predicted.to_sympy(item.info.variables), item.info.numbers)
                correct = not exception and self._checker.check_answer(item.info.answers, result)
                result_dump_b['answer_generated'] = [{k: str(v) for k, v in res.items()}
                                                     for res in result]

            # (2-4) Register metrics
            eqn_metrics['correct'].append(correct)
            result_dump_b['correct'] = correct

            result_dumps.append(result_dump_b)

        
        mwp_keys = mwp_references.keys()
        mwp_hypotheses = {key: mwp_hypotheses.get(key, ['']) for key in mwp_keys}

        metric: dict = {key: float(value)
                        for key, value in compute_metrics(mwp_references, mwp_hypotheses).items()}

        for key, items in eqn_metrics.items():
            metric[key] = sum(items) / len(items)

        if tokenizer is not None:
            metric['dump'] = result_dumps
        return metric

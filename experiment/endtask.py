import abc

from numpy.random import PCG64, Generator

from common.data import Example, Explanation
from common.dataset import Dataset
from experiment.base import ExperimentBase, ExperimentWithCorrectnessChange, ExperimentWithTreeChange
from model import MathWordProblemGenerator


class ErrorPropagationExperiment(ExperimentBase, abc.ABC):
    def __init__(self, module: MathWordProblemGenerator, dataset: Dataset, batch_size: int = 4, **solver_args):
        super().__init__(module, dataset, batch_size, **solver_args)
        self._rng = Generator(PCG64(1))

    def _prepare_experiment(self, **kwargs):
        self._rng = Generator(PCG64(1))

    def _transform_to_experiment_group(self, batch: Example) -> dict:
        # No modification for the batch
        batch = batch.to(self._module.device)
        mwp = []
        for expl in batch.explanation:
            rng_choice = self._rng.choice(len(expl.numbers))
            mwp.append(Explanation([expl.numbers[rng_choice]],
                                           [expl.variables[rng_choice]], worker=0))

        return dict(text=batch.text, mwp=mwp, dont_generate_expl=True)


class CorrectnessErrorPropagationExperiment(ErrorPropagationExperiment, ExperimentWithCorrectnessChange):
    pass


class TreeErrorPropagationExperiment(ErrorPropagationExperiment, ExperimentWithTreeChange):
    pass

from .faithfulness import SufficiencyErasureExperiment, ComprehensivenessErasureExperiment
from .endtask import CorrectnessErrorPropagationExperiment, TreeErrorPropagationExperiment

ERROR_CORR = 'error_propagation_corr'
ERROR_TREE = 'error_propagation_tree'

EXPERIMENT_TYPES = {
    ERROR_CORR: CorrectnessErrorPropagationExperiment,
    ERROR_TREE: TreeErrorPropagationExperiment
}

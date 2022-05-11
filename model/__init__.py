from model.base.chkpt import CheckpointingModule
from .ept import EPT, MODEL_CLS
from .eptg import MathGenerator
from .swan import *

MODELS = {
    'EPT': EPT,
    'EPT-G': MathGenerator
}


def model_create(config: dict) -> EPT:
    instance = MODELS[config[MODEL_CLS]](**config)
    instance.eval()
    return instance


def model_load(path: str) -> EPT:
    for cls in MODELS.values():
        if cls.checkpoint_path(path).exists():
            instance = cls.create_or_load(path)
            instance.eval()
            return instance

    raise FileNotFoundError('I cannot find any checkpoint file from the specified directory %s' % path)

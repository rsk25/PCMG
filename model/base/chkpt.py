from pathlib import Path

import torch
from torch import nn
from common.const.model import MDL_Q_ENC

class CheckpointingModule(nn.Module):
    def __init__(self, **config):
        super().__init__()
        self.config = config

    @classmethod
    def checkpoint_path(cls, directory: str):
        return Path(directory, '%s.pt' % cls.__name__)

    @classmethod
    def create_or_load(cls, path: str = None, **config):
        state = None

        if path is not None and cls.checkpoint_path(path).exists():
            print("Loading from pretrained.")
            with cls.checkpoint_path(path).open('rb') as fp:
                load_preset = torch.load(fp)

            new_config = {}
            new_config[MDL_Q_ENC] = config[MDL_Q_ENC]
            new_config.update(load_preset['config'])
            state = load_preset['state']

            model = cls(**new_config)
        else:
            model = cls(**config)
            
        if state is not None:
            print("State is not None")
            new_state = state.copy()
            for key in state.keys():
                if 'equation' in key:
                    new_state.pop(key)
                    new_state[key.replace('equation.','')] = state[key]
            model.load_state_dict(state, strict=False)

        return model

    def save(self, directory: str):
        with self.checkpoint_path(directory).open('wb') as fp:
            torch.save({
                'config': self.config,
                'state': self.state_dict()
            }, fp)


__all__ = ['CheckpointingModule']

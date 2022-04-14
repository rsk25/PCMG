from typing import Tuple, List, Optional, Any, Callable

import torch

from model.swan import SWANPhase1Only
from common.data import Text, Equation, Explanation, Encoded, EquationPrediction, Label
from common.data.base import move_to


class NVIXVisualizer(SWANPhase1Only):
    def __init__(self, **config):
        super.__init__(**config)


    

__all__ = ['NVIXVisualizer']
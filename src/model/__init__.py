from .base_trainer import BaseTrainer
from .mvgrl import *
from .mvgrl_spectral import Garner_Trainer
from .utils import *


def choose_trainer(model_name: str) -> BaseTrainer:
    if model_name == 'mvgrl':
        return MVGRLTrainer()
    if model_name == 'mvgrl_sviaug':
        return MVGRL_SVIAugTrainer()
    if model_name == 'garner':
        return Garner_Trainer()
    if model_name == 'garner_shuf':
        return Garner_Trainer()
    else:
        raise ValueError(f'Unknown model name: {model_name}')

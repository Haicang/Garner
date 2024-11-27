import os
from abc import ABC, abstractmethod
import time
import argparse

import numpy as np
import torch
import dgl

from ..utils import choose_device


class BaseTrainer(ABC):
    @staticmethod
    def get_parser():
        """Provide some arguments that are common to all trainers."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='singapore')
        parser.add_argument('--data-dir', type=str, default='data')
        parser.add_argument('--match-distance', type=int, default=50,
                            help='Distance threshold to match SVI to road. (Choose from [30, 40, 50].)')
        parser.add_argument('--gpu', type=int, default=2, help='-1 for cpu')
        parser.add_argument('--seed', type=int, default=-1, 
                            help='Random seed. Negative for random seed.')
        parser.add_argument('--road-feat-only', action='store_true', default=False,
                            help='''Whether to use road feature only.''')
        parser.add_argument('--epochs', type=int, default=500,
                            help='Number of epochs to train.')
        parser.add_argument('--proj-dim-a', type=int, default=256,
                            help='Dimension of the road feature projection layer.')
        parser.add_argument('--proj-dim-b', type=int, default=256,
                            help='Dimension of the svi embedding projection layer.')
        parser.add_argument('--emb-filename', type=str, default=None,
                            help='Filename to save the embedding.')
        parser.add_argument('--test-ratio', type=float, default=0.2,
                            help='Ratio of test set.')
        return parser
    
    def set_env(self, args):
        """Set the environment."""
        if args.seed >= 0:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            dgl.seed(args.seed)
        self.device = choose_device(args.gpu)
    
    @abstractmethod
    def get_args(self):
        """For each trainer, get the parser from `get_parser` and parse the arguments."""
        pass

    @abstractmethod
    def set_args(self, args):
        """Set the arguments."""
        pass

    @abstractmethod
    def data_process(self):
        """Process the data."""
        pass

    @abstractmethod
    def train(self):
        """Train the model."""
        pass

    @staticmethod
    def add_sim_parameters(parser):
        """Add parameters related to similarity graph."""
        parser.add_argument('--sim-type', type=str, default='knn', choices=['knn', 'threshold', 'mutual-knn', 'sparsified'],
                            help='Type of similarity graph.')
        parser.add_argument('--sim-k', type=int, default=6, help='Hyper-parameter for kNN graph.')
        parser.add_argument('--sim-epsilon', type=float, default=0.2, help='Hyper-parameter for threshold graph.')
        parser.add_argument('--sim-sigma', type=float, default=1., help='Hyper-parameter for threshold graph.')

        return parser

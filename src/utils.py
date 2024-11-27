import os
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import scipy.sparse as sp
import dgl


class EarlyStopper(object):
    """
    Warnings : Please specify the ``fname`` if you want to run
        multiple models simultaneously.

    Parameters
    ----------
    patience : int
    in_memory : bool
        Save the state_dict of the model in memory or on disk
    fname : str
        If save state_dict on disk, it's the filename

    """
    def __init__(self, mode='min', patience=10, in_memory=True,
                 fname='checkpoint.pt'):
        assert mode in ['min', 'max']
        self.mode = mode
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.in_memory = in_memory
        self.state_dict = None
        self.write_name = fname
        self._msg = None
        if mode == 'min':
            self.compare = lambda x, y: x < y
        else:
            self.compare = lambda x, y: x > y

    def step(self, score, model):
        """
        Parameters
        ----------
        score : float
        model : torch.nn.Module

        Returns
        -------
        early_stop : bool
            Whether to early stop or not.
        msg: str
            The message to print.
        """
        self._msg = None
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.compare(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            self._msg = f'EarlyStopping counter: {self.counter} out of {self.patience}'
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
    
    @property
    def msg(self):
        """The message to print for each step."""
        return self._msg

    def save_checkpoint(self, model):
        """Saves model when validation loss decrease."""
        if self.in_memory:
            # The `in_memory` version is slightly slower,
            # but it does NOT require a lot of writes to the disk.
            self.state_dict = deepcopy(model.state_dict())
        else:
            torch.save(model.state_dict(), self.write_name)

    def load_checkpoint(self):
        if self.in_memory:
            rst = self.state_dict
        else:
            rst = torch.load(self.write_name)
        return rst
    

@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Parameters
    ----------
    logits : torch.Tensor, shape [n_nodes, n_classes]
    labels : torch.Tensor, shape [n_nodes]

    Returns
    -------
    acc : float
    """
    preds = torch.argmax(logits, dim=1)
    correct = torch.sum(preds == labels).float()
    return correct.cpu().item() / labels.shape[0]


@torch.no_grad()
def mean_absolute_relative_error(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    are = torch.abs(preds - target) / target.abs()
    return are.mean()


# def dgl_from_sparsematrix(adj: dgl.sparse.SparseMatrix, eweight='weight') -> dgl.DGLGraph:
#     g = dgl.graph()


def normalize_adj(adj: sp.spmatrix) -> sp.spmatrix:
    """Symmetrically normalize adjacency matrix."""
    if not isinstance(adj, sp.coo_matrix):
        adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# def normalize_adj_dglsp(sm: dgl.sparse.SparseMatrix) -> dgl.sparse.SparseMatrix:
#     rowsum = sm.sum(1)
#     d_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
#     d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = torch.sparse.diag(d_inv_sqrt)
#     return (sm.t @ d_mat_inv_sqrt).t @ d_mat_inv_sqrt


def normalize_adj_dgl_graph(g: dgl.DGLGraph, eweight='weight', mode='sp') -> dgl.DGLGraph:
    assert mode in ['sp', 'mp']
    if mode=='sp':
        adj = g.adj_external(transpose=False, scipy_fmt='coo')
        adj.data = g.edata[eweight].cpu().numpy().squeeze()
        nadj = normalize_adj(adj)
        g = dgl.from_scipy(nadj, eweight_name=eweight)
        g.edata[eweight] = g.edata[eweight].to(torch.float32).resize_((g.num_edges(), 1))
    else:
        raise NotImplementedError(f'Unsupported mode: {mode}')
    return g


def laplacian_matrix(adj: sp.spmatrix, normalize=None) -> sp.spmatrix:
    if not isinstance(adj, sp.coo_matrix):
        adj = sp.coo_matrix(adj)
    if normalize is None:
        rowsum = np.array(adj.sum(1)).squeeze()
        laplacian = sp.diags(rowsum) - adj
    elif normalize == 'sym':
        nadj = normalize_adj(adj)
        laplacian = sp.eye(nadj.shape[0]) - nadj
    else:
        raise ValueError(f'Unsupported normalize type: {normalize}')
    return laplacian


@DeprecationWarning
def laplacian_matrix_dgl_graph(g: dgl.DGLGraph, eweight='weight', normalize=None) -> dgl.DGLGraph:
    """
    Assume the g is a directed graph or undirected (bidirected) graph.
    """
    adj = g.adj_external(transpose=True, scipy_fmt='coo')
    if len(g.edata.keys()) == 0:
        adj.data = np.ones((g.num_edges(), ), dtype=np.float32)
    else:
        adj.data = g.edata[eweight].cpu().numpy().squeeze()
    lap = laplacian_matrix(adj, normalize)
    g = dgl.from_scipy(lap, eweight_name=eweight)
    g.edata[eweight] = g.edata[eweight].to(torch.float32).resize_((g.num_edges(), 1))
    return g


###########################
# Device Helper Functions #
###########################
def choose_device(gpu_id: int) -> torch.device:
    assert isinstance(gpu_id, int)
    if not torch.cuda.is_available():
        gpu_id = -1
        warnings.warn("CUDA is not available, use CPU instead.")
    if gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', gpu_id)
    return device


def load_cached_param(param_name: str, cache_dir: str) -> torch.Tensor:
    """
    Parameters
    ----------
    param_name : str
    cache_dir : str

    Returns
    -------
    param : torch.Tensor
    """
    # Check if the cache directory exists.
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    # Check if the cache file exists.
    cache_file = os.path.join(cache_dir, param_name)
    if os.path.exists(cache_file):
        param = torch.load(cache_file)
    else:
        param = None
    return param


@torch.no_grad()
def memory_size(ts: torch.Tensor):
    size = ts.element_size() * ts.nelement() / 8 / 1024 / 1024
    if size < 1024:
        return f'{size:.2f} MB'
    else:
        return f'{size / 1024:.2f} GB'


def check_file_dir_exists(filepath: str):
    assert '/' in filepath
    dirpath = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

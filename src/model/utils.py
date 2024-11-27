import os
import torch
import torch.nn as nn
import dgl
import dgl.sparse
from ..constant import PROJECT_DIR


__all__ = ["FeatureProjection",
           "FeatureProjectionSingle",
           "laplacian_trace",
           "load_svi_similarity_graph",
           "gaussian_kernel",
           "threshold_graph"]


@torch.no_grad()
def gaussian_kernel(X: torch.Tensor, Y: torch.Tensor, sigma=1., chunk_size=1024):
    """Compute the Gaussian kernel between two matrices.
    Each line is a sample. Remember to use GPU if possible.

    X: torch.Tensor, shape [n, d]
    Y: torch.Tensor, shape [m, d]
    """
    assert X.shape[1] == Y.shape[1]
    n_feat = X.shape[1]
    n, m = X.shape[0], Y.shape[0]
    if n * m > chunk_size ** 2:
        # Compute the kernel in chunks to save memory.
        K = torch.zeros(n, m)
        for i in range(0, n, chunk_size):
            for j in range(0, m, chunk_size):
                i_end = min(i + chunk_size, n)
                j_end = min(j + chunk_size, m)
                K[i:i_end, j:j_end] = gaussian_kernel(X[i:i_end], Y[j:j_end], sigma=sigma)
        return K
    X = X.unsqueeze(1).expand(n, m, n_feat)
    Y = Y.unsqueeze(0).expand(n, m, n_feat)
    return torch.exp(-torch.sum((X - Y) ** 2, dim=2) / (2 * sigma ** 2))


@torch.no_grad()
def euclidean_dist(X: torch.Tensor, Y: torch.Tensor, chunk_size=1000):
    assert X.shape[1] == Y.shape[1]
    n_feat = X.shape[1]
    n, m = X.shape[0], Y.shape[0]
    if n * m > chunk_size ** 2:
        # Compute the kernel in chunks to save memory.
        K = torch.zeros(n, m)
        for i in range(0, n, chunk_size):
            for j in range(0, m, chunk_size):
                i_end = min(i + chunk_size, n)
                j_end = min(j + chunk_size, m)
                K[i:i_end, j:j_end] = euclidean_dist(X[i:i_end], Y[j:j_end])
        return K
    X = X.unsqueeze(1).expand(n, m, n_feat)
    Y = Y.unsqueeze(0).expand(n, m, n_feat)
    return torch.sum((X - Y) ** 2, dim=2) ** 0.5


@torch.no_grad()
def cosine_similarity(X: torch.Tensor, Y: torch.Tensor, chunk_size=1000):
    assert X.shape[1] == Y.shape[1]
    n_feat = X.shape[1]
    n, m = X.shape[0], Y.shape[0]
    if n * m > chunk_size ** 2:
        # Compute the kernel in chunks to save memory.
        K = torch.zeros(n, m)
        for i in range(0, n, chunk_size):
            for j in range(0, m, chunk_size):
                i_end = min(i + chunk_size, n)
                j_end = min(j + chunk_size, m)
                K[i:i_end, j:j_end] = cosine_similarity(X[i:i_end], Y[j:j_end])
        return K
    X = X.unsqueeze(1).expand(n, m, n_feat)
    Y = Y.unsqueeze(0).expand(n, m, n_feat)
    return torch.sum(X * Y, dim=2) / (torch.norm(X, dim=2) * torch.norm(Y, dim=2))


@torch.no_grad()
def threshold_graph(X: torch.Tensor, epsilon: float, sigma: float = 1.) -> dgl.DGLGraph:
    """Compute the threshold graph for the given data.
    Each line is a sample. Remember to use GPU if possible.

    X: torch.Tensor, shape [n, d]
    epsilon: float, threshold
    sigma: float, bandwidth of the Gaussian kernel
    """
    K = gaussian_kernel(X, X, sigma=sigma)
    K[K < epsilon] = 0
    adj = K.to_sparse_coo()
    adj = adj.coalesce()
    g = dgl.graph((adj.indices()[0], adj.indices()[1]), num_nodes=X.shape[0])
    g.edata['weight'] = adj.values()
    return g


def load_svi_similarity_graph(dataname:str, svi_emb: torch.Tensor, type: str, 
                              k: int, epsilon: float, sigma: float = 1., device='cpu'):
    """
    This function build a similarity graph based on the SVI embeddings on roads.
    The similarity graph is either a kNN graph or a threshold graph.
    The graph is saved in the cache folder.
    Thus this function will load the graph after the first time it is called.
    """
    # Preparing the setting
    assert type in ['knn', 'threshold', 'mutual-knn', 'sparsified']
    cache_path = os.path.join(PROJECT_DIR, 'cache', 'similarity_graph')
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if type == 'knn':
        cache_name = f'{dataname}_knn_{k}.bin'
    elif type == 'threshold':
        cache_name = f'{dataname}_threshold_{epsilon}_{sigma}.bin'
    elif type == 'sparsified':
        cache_path = os.path.join(PROJECT_DIR, 'cache', 'sparsified_graph')
        cache_name = f'{dataname}_sim_sparsified_graph.bin'
    else:
        raise ValueError(f'Unknown type: {type}')
    
    # Load the graph if it exists.
    cache_graph_path = os.path.join(cache_path, cache_name)
    if os.path.exists(cache_graph_path):
        graph = dgl.load_graphs(cache_graph_path)[0][0]
        if type == 'sparsified':
            graph = dgl.add_self_loop(graph)
    else:
        if device in ['gpu', 'cuda']:
            device = torch.device('cuda')
        assert isinstance(device, torch.device)
        svi_emb = svi_emb.to(device)
        if type == 'knn':
            graph = dgl.knn_graph(svi_emb, k, 'bruteforce', exclude_self=True)
            # Do not use `bruteforce-blas` due to its huge memory consumption.
        elif type == 'threshold':
            graph = threshold_graph(svi_emb, epsilon, sigma)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph = graph.cpu()
        svi_emb = svi_emb.cpu()
        dgl.save_graphs(cache_graph_path, [graph])
        torch.cuda.empty_cache()
    return graph


def laplacian_trace(g:dgl.DGLGraph, feat: torch.Tensor, eweight='weight', mode='sp') -> torch.Tensor:
    """
    Compute the trace of graph Laplacian matrix of graph `g`, with `feat` as the node features.
    Tr(X^T L X)
    Graph `g` has been preprocessed to have self-loop and bidirection links.

    mode: `mp` indicates dgl message passing, while `sp` indicates dgl sparse matrix multiplication.
    """
    assert mode in ['mp', 'sp']
    if mode == 'mp':
        with g.local_scope():
            g.ndata['ft'] = feat
            g.apply_edges(dgl.function.u_sub_v('ft', 'ft', 'm'))
            if eweight in g.edata:
                return (g.edata['m'] * g.edata['m'] * g.edata[eweight]).sum() * 0.5
            else:
                return (g.edata['m'] * g.edata['m']).sum() * 0.5
    else:
        with g.local_scope():
            A = g.adj()
            if eweight in g.edata:
                A.values = g.edata[eweight]
            rowsum = A.sum(dim=1)
            L = dgl.sparse.diag(rowsum) - A
            return ((L.T @ feat) * feat).sum()


class FeatureProjection(nn.Module):
    def __init__(self, in_dim_a, in_dim_b, out_dim_a, out_dim_b):
        super().__init__()
        self.proj_a = nn.Linear(in_dim_a, out_dim_a)
        self.proj_b = nn.Linear(in_dim_b, out_dim_b)

    def forward(self, a, b):
        return self.proj_a(a), self.proj_b(b)
    
    def forward_concat(self, a, b):
        a, b = self.forward(a, b)
        return torch.cat([a, b], dim=1)
    

class FeatureProjectionSingle(nn.Module):
    """
    This class has the same interface as FeatureProjection, 
    but it only projects the first input. 
    It is used for baselines that only use the road features only without SVI.
    """
    def __init__(self, in_dim_a, in_dim_b, out_dim_a, out_dim_b):
        super().__init__()
        self.proj_a = nn.Linear(in_dim_a, out_dim_a)

    def forward(self, a, b):
        return self.proj_a(a)
    
    def forward_concat(self, a, b):
        return self.forward(a, b)

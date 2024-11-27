"""
The code is adapted from https://github.com/hengruizhang98/mvgrl, 
which is also collected in dgl-examples/pytorch.
"""

import os
import argparse
import random
import time

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv, APPNPConv
from dgl.nn.pytorch.glob import AvgPooling

from src.constant import CACHE_DIR

from ..utils import EarlyStopper, normalize_adj
from .utils import *
from ..constant import *
from .base_trainer import BaseTrainer


class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.fn = nn.Bilinear(dim, dim, 1)

    def forward(self, h1, h2, h3, h4, c1, c2):
        c_x1 = c1.expand_as(h1).contiguous()
        c_x2 = c2.expand_as(h2).contiguous()

        # positive
        sc_1 = self.fn(h2, c_x1).squeeze(1)
        sc_2 = self.fn(h1, c_x2).squeeze(1)

        # negative
        sc_3 = self.fn(h4, c_x1).squeeze(1)
        sc_4 = self.fn(h3, c_x2).squeeze(1)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4))

        return logits
    

class MVGRL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MVGRL, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        self.pooling = AvgPooling()

        self.disc = Discriminator(out_dim)
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, feat, edge_weight):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        return (h1 + h2).detach()

    def forward(self, graph, diff_graph, feat, shuf_feat, edge_weight, output_emb=False):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)

        h3 = self.encoder1(graph, shuf_feat)
        h4 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))

        out = self.disc(h1, h2, h3, h4, c1, c2)

        if output_emb:
            return out, (h1 + h2)
        return out
    

class MVGRLWithProjection(nn.Module):
    """
    This class can handle both road feature only and road feature + SVI embedding.
    """
    def __init__(self, in_dim_a, in_dim_b, proj_dim_a, proj_dim_b, 
                 out_dim, road_feat_only=False):
        super().__init__()
        self.road_feat_only = road_feat_only
        if road_feat_only:
            self.proj = FeatureProjectionSingle(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.mvgrl = MVGRL(proj_dim_a, out_dim)
        else:
            self.proj = FeatureProjection(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.mvgrl = MVGRL(proj_dim_a + proj_dim_b, out_dim)

    def get_embedding(self, graph, diff_graph, feat_a, feat_b, edge_weight):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        return self.mvgrl.get_embedding(graph, diff_graph, proj_feat, edge_weight)

    def forward(self, graph, diff_graph, feat_a, feat_b, shuf_feat_a, shuf_feat_b, edge_weight, output_emb=False):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        proj_shuf_feat = self.proj.forward_concat(shuf_feat_a, shuf_feat_b)
        return self.mvgrl(graph, diff_graph, proj_feat, proj_shuf_feat, edge_weight, output_emb)


class MVGRLTrainer(BaseTrainer):
    def __init__(self) -> None:
        super().__init__()
        self.args: argparse.Namespace = None
        self.device: torch.device = None

    @staticmethod
    def get_args(argv_list=None):
        parser = MVGRLTrainer.get_parser()
        parser.add_argument('--patience', type=int, default=999, 
                            help='Patient epochs to wait before early stopping. 0 for no early stopping.')
        parser.add_argument('--k', type=int, default=20, help='Number of propagation steps for graph diffusion.')
        parser.add_argument('--alpha', type=float, default=0.2, help='Propagation factor for graph diffusion.')
        parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
        parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
        parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
        parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
        parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
        parser.add_argument("--hid-dim", type=int, default=512, help='Hidden layer dim.')
        parser.add_argument("--sample-size", type=int, default=4000, help='Subgraph size.')

        if argv_list is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv_list)
        return args

    def set_args(self, args):
        self.args = args
        print(args)

    def data_process(self, data, cache_dir=CACHE_DIR + '/mvgrl'):
        """
        Get the diffusion graph with appnp approximation.
        """
        epsilon = self.args.epsilon
        k = self.args.k
        alpha = self.args.alpha
        if cache_dir is not None:
            cache_dir = os.path.join(cache_dir, self.args.dataset)
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            diff_graph_path = os.path.join(cache_dir, f'diff_graph_k{k}_alpha{alpha}_eps{epsilon}.bin')
            diff_weight_path = os.path.join(cache_dir, f'diff_weight_k{k}_alpha{alpha}_eps{epsilon}.pt')
            if os.path.exists(diff_graph_path) and os.path.exists(diff_weight_path):
                diff_graph = dgl.load_graphs(diff_graph_path, [0])[0][0]
                diff_weight = torch.load(diff_weight_path, map_location=torch.device('cpu'))
                return {'diff_graph': diff_graph, 'diff_weight': diff_weight}
        graph = data.g
        # Self loop has been added during the dataset transform.
        appnp = APPNPConv(k, alpha)
        id = torch.eye(graph.number_of_nodes(), dtype=torch.float)
        # TODO: use gpu to speed up the following data process.
        # The preprocessing time is about 2min on CPU, so I do not think GPU is required.
        # Also the computation may use too much memory.
        time_start = time.time()
        diff_adj = appnp(graph, id).numpy()
        time_end = time.time()
        # print(f'Diffusion time: {time_end - time_start}')
        diff_adj[diff_adj < epsilon] = 0

        scaler = MinMaxScaler()
        scaler.fit(diff_adj)
        diff_adj = scaler.transform(diff_adj)
        diff_edges = np.nonzero(diff_adj)
        diff_weight = diff_adj[diff_edges]
        diff_weight = torch.from_numpy(diff_weight).float()
        diff_graph = dgl.graph(diff_edges)
        time_c = time.time()
        print(f'Diffusion time: {time_end - time_start} (+ {time_c - time_end}) (s)')
        if cache_dir is not None:
            dgl.save_graphs(diff_graph_path, [diff_graph])
            torch.save(diff_weight, diff_weight_path)
        return {'diff_graph': diff_graph, 'diff_weight': diff_weight}

    def train(self, data, **kwargs):
        args = self.args
        device = self.device
        graph = data.g
        road_feat = data.road_feat
        svi_emb = data.svi_emb
        diff_graph = kwargs['diff_graph']
        edge_weight = kwargs['diff_weight']
        diff_graph.edata['edge_weight'] = edge_weight

        n_node = graph.number_of_nodes()
        sample_size = args.sample_size
        lbl1 = torch.ones(sample_size * 2)
        lbl2 = torch.zeros(sample_size * 2)
        lbl = torch.cat((lbl1, lbl2))
        lbl = lbl.to(device)

        # Step 2: Create model and training components
        model = MVGRLWithProjection(road_feat.shape[1], svi_emb.shape[1], 
                                    args.proj_dim_a, args.proj_dim_b, args.hid_dim, 
                                    road_feat_only=args.road_feat_only).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, 
                                     weight_decay=args.wd1)
        criterion = nn.BCEWithLogitsLoss()
        node_list = list(range(n_node))
        stopper = EarlyStopper(patience=args.patience)

        # Step 3: Training
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            sample_idx = random.sample(node_list, sample_size)
            sample_idx = torch.LongTensor(sample_idx)
            g = dgl.node_subgraph(graph, sample_idx)
            dg = dgl.node_subgraph(diff_graph, sample_idx)

            feat_a = road_feat[sample_idx]
            feat_b = svi_emb[sample_idx]
            ew = dg.edata.pop('edge_weight')
            shuf_idx = torch.randperm(sample_size)
            shuf_feat_a = feat_a[shuf_idx, :]
            shuf_feat_b = feat_b[shuf_idx, :]

            g = g.to(device)
            dg = dg.to(device)
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            ew = ew.to(device)
            shuf_feat_a = shuf_feat_a.to(device)
            shuf_feat_b = shuf_feat_b.to(device)

            logits = model(g, dg, feat_a, feat_b, shuf_feat_a, shuf_feat_b, ew)
            loss = criterion(logits, lbl)
            loss.backward()
            optimizer.step()

            if args.patience > 0 and stopper.step(loss.item(), model):
                print('Early stop!')
                break

            early_stop_msg = stopper.msg
            msg = f'Epoch: {epoch} | Loss: {loss.item():0.4f}'
            if early_stop_msg is not None:
                msg = msg + ' | ' + early_stop_msg
            print(msg)

        # Step 4: Get embedding
        model.eval()
        model.load_state_dict(stopper.load_checkpoint())
        graph = graph.to(device)
        diff_graph = diff_graph.to(device)
        feat_a = road_feat.to(device)
        feat_b = svi_emb.to(device)
        edge_weight = edge_weight.to(device)
        emb = model.get_embedding(graph, diff_graph, feat_a, feat_b, edge_weight)
        return emb.cpu().detach()


class MVGRL_LossTrainer(MVGRLTrainer):
    """
    Add 3rd law as an additional loss.
    """
    @staticmethod
    def get_args(argv_list=None):
        parser = MVGRLTrainer.get_parser()
        parser.add_argument('--patience', type=int, default=999, 
                            help='Patient epochs to wait before early stopping. 0 for no early stopping.')
        parser.add_argument('--k', type=int, default=20, help='Number of propagation steps for graph diffusion.')
        parser.add_argument('--alpha', type=float, default=0.2, help='Propagation factor for graph diffusion.')
        parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
        parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
        parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
        parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
        parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
        parser.add_argument("--hid-dim", type=int, default=512, help='Hidden layer dim.')
        parser.add_argument("--sample-size", type=int, default=4000, help='Subgraph size.')
        parser.add_argument("--sparsified-graph", type=str, default=None, help='filename of the sparsified graph.')
        parser.add_argument("--beta", type=float, default=1, help='Weight of the loss of the 3rd law.')

        if argv_list is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv_list)
        return args
    
    def data_process(self, data, cache_dir=CACHE_DIR, eweight='weight'):
        # Load diffused graph for mvgrl.
        processed_data = super().data_process(data, cache_dir + '/mvgrl') 
        assert isinstance(processed_data, dict)
        # Load sparsified sim graph for the 3rd law.
        sparsified_sim_graph: dgl.DGLGraph = dgl.load_graphs(cache_dir + '/sparsified_graph/' + self.args.sparsified_graph)[0][0]
        adj = sparsified_sim_graph.adj_external(scipy_fmt='coo')
        adj.data = sparsified_sim_graph.edata[eweight].numpy().squeeze()
        adj = adj + adj.T
        norm_adj = normalize_adj(adj)
        g = dgl.from_scipy(norm_adj, eweight_name=eweight)
        g.edata[eweight] = g.edata[eweight].to(torch.float32).resize_((g.num_edges(), 1))
        processed_data['sparsified_sim_graph'] = g
        return processed_data

    def train(self, data, **kwargs):
        args = self.args
        device = self.device
        graph = data.g
        road_feat = data.road_feat
        svi_emb = data.svi_emb
        diff_graph = kwargs['diff_graph']
        edge_weight = kwargs['diff_weight']
        diff_graph.edata['edge_weight'] = edge_weight
        sparsified_sim_graph = kwargs['sparsified_sim_graph']

        n_node = graph.number_of_nodes()
        sample_size = args.sample_size
        lbl1 = torch.ones(sample_size * 2)
        lbl2 = torch.zeros(sample_size * 2)
        lbl = torch.cat((lbl1, lbl2))
        lbl = lbl.to(device)

        # Step 2: Create model and training components
        model = MVGRLWithProjection(road_feat.shape[1], svi_emb.shape[1], 
                                    args.proj_dim_a, args.proj_dim_b, args.hid_dim, 
                                    road_feat_only=args.road_feat_only).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, 
                                     weight_decay=args.wd1)
        criterion = nn.BCEWithLogitsLoss()
        node_list = list(range(n_node))
        stopper = EarlyStopper(patience=args.patience)

        # Step 3: Training
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            sample_idx = random.sample(node_list, sample_size)
            sample_idx = torch.LongTensor(sample_idx)
            g = dgl.node_subgraph(graph, sample_idx)
            dg = dgl.node_subgraph(diff_graph, sample_idx)
            ssg = dgl.node_subgraph(sparsified_sim_graph, sample_idx)

            feat_a = road_feat[sample_idx]
            feat_b = svi_emb[sample_idx]
            ew = dg.edata.pop('edge_weight')
            shuf_idx = torch.randperm(sample_size)
            shuf_feat_a = feat_a[shuf_idx, :]
            shuf_feat_b = feat_b[shuf_idx, :]

            g = g.to(device)
            dg = dg.to(device)
            ssg = ssg.to(device)
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            ew = ew.to(device)
            shuf_feat_a = shuf_feat_a.to(device)
            shuf_feat_b = shuf_feat_b.to(device)

            logits, emb = model(g, dg, feat_a, feat_b, shuf_feat_a, shuf_feat_b, ew, output_emb=True)
            loss1 = criterion(logits, lbl)
            loss2 = args.beta * laplacian_trace(ssg, emb) / (emb.shape[1] * emb.shape[0])
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch} | Loss: {loss.item():0.4f} | Loss1: {loss1.item():0.4f} | Loss2: {loss2.item():0.4f}')
            if args.patience > 0 and stopper.step(loss.item(), model):
                break

        # Step 4: Get embedding
        model.eval()
        graph = graph.to(device)
        diff_graph = diff_graph.to(device)
        feat_a = road_feat.to(device)
        feat_b = svi_emb.to(device)
        edge_weight = edge_weight.to(device)
        emb = model.get_embedding(graph, diff_graph, feat_a, feat_b, edge_weight)
        return emb.cpu().detach()
    

class MVGRL_SVIAug(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MVGRL_SVIAug, self).__init__()

        self.encoder1 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.encoder2 = GraphConv(in_dim, out_dim, norm='none', bias=True, activation=nn.PReLU())
        self.encoder3 = GraphConv(in_dim, out_dim, norm='both', bias=True, activation=nn.PReLU())
        self.pooling = AvgPooling()

        self.disc1 = Discriminator(out_dim)
        self.disc2 = Discriminator(out_dim)
        # May use a different discriminator in the future.
        self.act_fn = nn.Sigmoid()

    def get_embedding(self, graph, diff_graph, sim_graph, feat, edge_weight, sim_weight=None):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)
        h3 = self.encoder3(sim_graph, feat, edge_weight=sim_weight)
        return (h1 + h2 + h3).detach()
    
    def forward(self, graph, diff_graph, sim_graph, feat, shuf_feat, edge_weight, sim_weight=None):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)
        h3 = self.encoder3(sim_graph, feat, edge_weight=sim_weight)

        h4 = self.encoder1(graph, shuf_feat)
        h5 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)
        h6 = self.encoder3(sim_graph, shuf_feat, edge_weight=sim_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))
        c3 = self.act_fn(self.pooling(graph, h3))

        # out = torch.cat((self.disc1(h1, h2, h4, h5, c1, c2), self.disc2(h1, h3, h4, h6, c1, c3)))
        out = (self.disc1(h1, h2, h4, h5, c1, c2), self.disc2(h1, h3, h4, h6, c1, c3))

        return out
    

class MVGRL_SVIAugWithProjection(nn.Module):
    def __init__(self, in_dim_a, in_dim_b, proj_dim_a, proj_dim_b, 
                 out_dim, road_feat_only=False):
        super().__init__()
        self.road_feat_only = road_feat_only
        if road_feat_only:
            self.proj = FeatureProjectionSingle(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.gcl = MVGRL_SVIAug(proj_dim_a, out_dim)
        else:
            self.proj = FeatureProjection(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.gcl = MVGRL_SVIAug(proj_dim_a + proj_dim_b, out_dim)
    
    def get_embedding(self, graph, diff_graph, sim_graph, feat_a, feat_b, edge_weight, sim_weight=None):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        return self.gcl.get_embedding(graph, diff_graph, sim_graph, proj_feat, edge_weight, sim_weight)

    def forward(self, graph, diff_graph, sim_graph, feat_a, feat_b, shuf_feat_a, shuf_feat_b, edge_weight, sim_weight=None):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        proj_shuf_feat = self.proj.forward_concat(shuf_feat_a, shuf_feat_b)
        return self.gcl(graph, diff_graph, sim_graph, proj_feat, proj_shuf_feat, edge_weight, sim_weight)


class MVGRL_SVIAugTrainer(MVGRLTrainer):
    @staticmethod
    def get_args(argv_list=None):
        parser = MVGRL_SVIAugTrainer.get_parser()
        parser.add_argument('--patience', type=int, default=999, 
                            help='Patient epochs to wait before early stopping. 0 for no early stopping.')
        parser.add_argument('--k', type=int, default=20, help='Number of propagation steps for graph diffusion.')
        parser.add_argument('--alpha', type=float, default=0.2, help='Propagation factor for graph diffusion.')
        parser.add_argument('--lr1', type=float, default=0.001, help='Learning rate of mvgrl.')
        parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate of linear evaluator.')
        parser.add_argument('--wd1', type=float, default=0., help='Weight decay of mvgrl.')
        parser.add_argument('--wd2', type=float, default=0., help='Weight decay of linear evaluator.')
        parser.add_argument('--epsilon', type=float, default=0.01, help='Edge mask threshold of diffusion graph.')
        parser.add_argument("--hid-dim", type=int, default=512, help='Hidden layer dim.')
        parser.add_argument("--sample-size", type=int, default=4000, help='Subgraph size.')
        parser = MVGRL_SVIAugTrainer.add_sim_parameters(parser)
        
        if argv_list is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv_list)
        return args
    
    def data_process(self, data, cache_dir=CACHE_DIR + '/mvgrl'):
        args = self.args
        processed_data = super().data_process(data, cache_dir)
        processed_data["sim_graph"] = load_svi_similarity_graph(args.dataset, data.svi_emb, 
                                                                args.sim_type, args.sim_k, 
                                                                args.sim_epsilon, args.sim_sigma, self.device)
        return processed_data

    def train(self, data, **kwargs):
        # TODO: support for sim_weight.
        args = self.args
        device = self.device
        graph = data.g
        road_feat = data.road_feat
        svi_emb = data.svi_emb
        diff_graph = kwargs['diff_graph']
        edge_weight = kwargs['diff_weight']
        diff_graph.edata['edge_weight'] = edge_weight
        sim_graph = kwargs['sim_graph']

        n_node = graph.number_of_nodes()
        sample_size = args.sample_size
        lbl1 = torch.ones(sample_size * 2)
        lbl2 = torch.zeros(sample_size * 2)
        lbl = torch.cat((lbl1, lbl2))
        lbl = lbl.to(device)

        model = MVGRL_SVIAugWithProjection(road_feat.shape[1], svi_emb.shape[1], 
                                           args.proj_dim_a, args.proj_dim_b, args.hid_dim, 
                                           road_feat_only=args.road_feat_only).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr1, 
                                     weight_decay=args.wd1)
        criterion = nn.BCEWithLogitsLoss()
        node_list = list(range(n_node))
        stopper = EarlyStopper(patience=args.patience)

        # Step 3: Training
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            sample_idx = random.sample(node_list, sample_size)
            sample_idx = torch.LongTensor(sample_idx)
            g = dgl.node_subgraph(graph, sample_idx)
            dg = dgl.node_subgraph(diff_graph, sample_idx)
            sg = dgl.node_subgraph(sim_graph, sample_idx)

            feat_a = road_feat[sample_idx]
            feat_b = svi_emb[sample_idx]
            ew = dg.edata.pop('edge_weight')
            shuf_idx = torch.randperm(sample_size)
            shuf_feat_a = feat_a[shuf_idx, :]
            shuf_feat_b = feat_b[shuf_idx, :]

            g = g.to(device)
            dg = dg.to(device)
            sg = sg.to(device)
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            ew = ew.to(device)
            shuf_feat_a = shuf_feat_a.to(device)
            shuf_feat_b = shuf_feat_b.to(device)

            logits = model(g, dg, sg, feat_a, feat_b, shuf_feat_a, shuf_feat_b, ew)
            logits1, logits2 = logits
            loss1 = criterion(logits1, lbl)
            loss2 = criterion(logits2, lbl)
            loss = 0.5 * loss1 + 0.5 * loss2
            # loss = criterion(logits, torch.cat((lbl, lbl)))
            loss.backward()
            optimizer.step()

            if args.patience > 0 and stopper.step(loss.item(), model):
                print('Early stop!')
                break

            early_stop_msg = stopper.msg
            msg = f'Epoch: {epoch} | Loss: {loss.item():0.4f}'
            if early_stop_msg is not None:
                msg = msg + ' | ' + early_stop_msg
            print(msg)

        # Step 4: Get embedding
        model.eval()
        model.load_state_dict(stopper.load_checkpoint())
        graph = graph.to(device)
        diff_graph = diff_graph.to(device)
        sim_graph = sim_graph.to(device)
        feat_a = road_feat.to(device)
        feat_b = svi_emb.to(device)
        edge_weight = edge_weight.to(device)
        emb = model.get_embedding(graph, diff_graph, sim_graph, feat_a, feat_b, edge_weight)
        return emb.cpu().detach()

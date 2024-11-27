import networkx as nx
from src.constant import CACHE_DIR
from .mvgrl import *


def load_svi_sim_neg_graph(dataname:str, n_nodes:int, d=20, index=None, cache_dir=CACHE_DIR, device='cpu'):
    cache_dir = os.path.join(cache_dir, 'sim_neg_graph')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if index is not None:
        dataname = f'{dataname}_{index}'
    cache_name = f'{dataname}_sim_neg_graph_{d}.bin'
    cache_path = os.path.join(cache_dir, cache_name)
    if os.path.exists(cache_path):
        return dgl.load_graphs(cache_path)[0][0]
    nxg = nx.random_regular_graph(d, n_nodes)
    g = dgl.from_networkx(nxg)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    dgl.save_graphs(cache_path, [g])
    return g


class Garner(MVGRL_SVIAug):
    def __init__(self, in_dim, out_dim, shuf_neg_feat=False):
        super().__init__(in_dim, out_dim)
        self.if_shuf_neg_feat = shuf_neg_feat
        
    def forward(self, graph, diff_graph, sim_graph, sim_neg_graph, feat, shuf_feat, edge_weight, sim_weight=None):
        h1 = self.encoder1(graph, feat)
        h2 = self.encoder2(diff_graph, feat, edge_weight=edge_weight)
        h3 = self.encoder3(sim_graph, feat, edge_weight=sim_weight)

        h4 = self.encoder1(graph, shuf_feat)
        h5 = self.encoder2(diff_graph, shuf_feat, edge_weight=edge_weight)
        if self.if_shuf_neg_feat:
            h6 = self.encoder3(sim_neg_graph, shuf_feat, edge_weight=sim_weight)
        else:
            h6 = self.encoder3(sim_neg_graph, feat, edge_weight=sim_weight)

        c1 = self.act_fn(self.pooling(graph, h1))
        c2 = self.act_fn(self.pooling(graph, h2))
        c3 = self.act_fn(self.pooling(graph, h3))

        out = (self.disc1(h1, h2, h4, h5, c1, c2), self.disc2(h1, h3, h4, h6, c1, c3))

        return out
    

class Garner_with_Projection(nn.Module):
    def __init__(self, in_dim_a, in_dim_b, proj_dim_a, proj_dim_b, 
                 out_dim, road_feat_only=False, shuf_neg_feat=False):
        super().__init__()
        self.road_feat_only = road_feat_only
        if road_feat_only:
            self.proj = FeatureProjectionSingle(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.gcl = Garner(proj_dim_a, out_dim, shuf_neg_feat)
        else:
            self.proj = FeatureProjection(in_dim_a, in_dim_b, proj_dim_a, proj_dim_b)
            self.gcl = Garner(proj_dim_a + proj_dim_b, out_dim, shuf_neg_feat)
    
    def get_embedding(self, graph, diff_graph, sim_graph, feat_a, feat_b, edge_weight, sim_weight=None):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        return self.gcl.get_embedding(graph, diff_graph, sim_graph, proj_feat, edge_weight, sim_weight)

    def forward(self, graph, diff_graph, sim_graph, sim_neg_graph, feat_a, feat_b, shuf_feat_a, shuf_feat_b, edge_weight, sim_weight=None):
        proj_feat = self.proj.forward_concat(feat_a, feat_b)
        proj_shuf_feat = self.proj.forward_concat(shuf_feat_a, shuf_feat_b)
        return self.gcl(graph, diff_graph, sim_graph, sim_neg_graph, proj_feat, proj_shuf_feat, edge_weight, sim_weight)


class Garner_Trainer(MVGRL_SVIAugTrainer):
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
        parser.add_argument('--shuf-feat', action='store_true', help='Use shuffled feature for negative sampling.')
        parser.add_argument("--sample-size", type=int, default=4000, help='Subgraph size.')
        parser = MVGRL_SVIAugTrainer.add_sim_parameters(parser)
        parser.add_argument('--sim-neg-d', type=int, default=20, help='Degree of negative similarity graph.')
        parser.add_argument('--sim-neg-index', type=int, default=0)
        
        if argv_list is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(argv_list)
        return args

    def data_process(self, data, cache_dir=CACHE_DIR):
        args = self.args
        processed_data = super().data_process(data, cache_dir + '/mvgrl')
        sng = load_svi_sim_neg_graph(args.dataset, data.svi_emb.shape[0], d=args.sim_neg_d, 
                                     index=args.sim_neg_index, cache_dir=cache_dir, device=self.device)
        sng = dgl.to_bidirected(sng)
        processed_data['sim_neg_graph'] = sng
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
        sim_graph = kwargs['sim_graph']
        sim_neg_graph = kwargs['sim_neg_graph']

        n_node = graph.number_of_nodes()
        sample_size = args.sample_size
        lbl1 = torch.ones(sample_size * 2)
        lbl2 = torch.zeros(sample_size * 2)
        lbl = torch.cat((lbl1, lbl2))
        lbl = lbl.to(device)

        model = Garner_with_Projection(road_feat.shape[1], svi_emb.shape[1], 
                                           args.proj_dim_a, args.proj_dim_b, args.hid_dim, 
                                           road_feat_only=args.road_feat_only,
                                           shuf_neg_feat=args.shuf_feat).to(device)
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
            sng = dgl.node_subgraph(sim_neg_graph, sample_idx)

            feat_a = road_feat[sample_idx]
            feat_b = svi_emb[sample_idx]
            ew = dg.edata.pop('edge_weight')
            shuf_idx = torch.randperm(sample_size)
            shuf_feat_a = feat_a[shuf_idx, :]
            shuf_feat_b = feat_b[shuf_idx, :]

            g = g.to(device)
            dg = dg.to(device)
            sg = sg.to(device)
            sng = sng.to(device)
            feat_a = feat_a.to(device)
            feat_b = feat_b.to(device)
            ew = ew.to(device)
            shuf_feat_a = shuf_feat_a.to(device)
            shuf_feat_b = shuf_feat_b.to(device)

            logits = model(g, dg, sg, sng, feat_a, feat_b, shuf_feat_a, shuf_feat_b, ew)
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

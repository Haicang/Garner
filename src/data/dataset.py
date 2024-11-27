import os
import time

import numpy as np
import geopandas as gpd
import osmnx as ox
import torch
import dgl
from dgl import DGLError

from ..constant import *
from .preprocessing import osm_map_preprocess


class RoadDataset(object):
    def __init__(self, dataname, data_dir=DATA_DIR, use_svi=True, transform=True, **kwargs):
        """
        dataname: "singapore" or "nyc"
        data_dir: path to the root directory of the dataset.
        match_distance: distance threshold to match SVI to road. (Choose from [30, 40, 50].)

        ---
        Notes:
        1. It is problematic to save the network data with networkx with `edgelist` or `adjlist`,
            I recommend to build the graph from gpd.DataFrame.
        2. I recommend to use pre-processed data. Because there may be some unexpected errors due to osm updates.
        """
        assert dataname in ["singapore", "nyc"]
        self.dataname = dataname
        # Path to processed data.
        self.data_path = os.path.join(data_dir, dataname, "data_processed")
        self.transformed = transform
        self.use_svi = use_svi
        data_path = self.data_path
        match_distance = kwargs.get("match_distance", 50)

        # Load road network data.
        try:
            dgl_g = dgl.load_graphs(os.path.join(data_path, f"{dataname}_roadnet.bin"))[0][0]
        except DGLError:
            time_a = time.time()
            print("Loading data...")
            try:
                edges = gpd.read_file(os.path.join(data_path, "edges.geojson"))
                nodes = gpd.read_file(os.path.join(data_path, "nodes.geojson"))
                # road_feat = np.load(os.path.join(data_path, "road_feat.npz"))['arr_0']
                edges = edges.set_index(['u', 'v', 'key'])
                print("Data loaded from local pre-processed data.")
            except:
                nodes, edges = osm_map_preprocess(data_dir, dataname)
            time_b = time.time()
            print(f"Data loaded. Time: {time_b - time_a} (s)")
            # Build graph.
            nxg = ox.graph_from_gdfs(nodes, edges)
            dgl_g = dgl.from_networkx(nxg)
            dgl.save_graphs(os.path.join(data_path, f"{dataname}_roadnet.bin"), [dgl_g])

        road_feat = np.load(os.path.join(data_path, "road_feat.npz"))['arr_0']
        road_feat = torch.from_numpy(road_feat).float()
        if use_svi:
            svi_emb = torch.load(os.path.join(data_path, f"svi_emb/svi_emb_on_road_{match_distance}.pt"))

        time_a = time.time()
        self.g = dgl_g
        self.road_feat = road_feat
        if use_svi:
            self.svi_emb = svi_emb
        # TODO: any advanced imputation method?
        if self.transformed:
            self.transform()
        time_b = time.time()
        print(f"Data pre-processed. Time: {time_b - time_a} (s)")

    def transform(self):
        self.g = dgl.line_graph(self.g)
        assert self.g.num_nodes() == self.road_feat.shape[0]
        self.add_self_loop()
        if self.use_svi:
            self.svi_impute()

    def add_self_loop(self):
        self.g = dgl.add_self_loop(self.g)

    def svi_impute(self):
        """
        Impute SVI embeddings on the road network. 
        Some of rows of svi_emb are zero vectors, impute them with the mean of the non-zero vectors.
        """
        svi_emb = self.svi_emb
        zero_rows = torch.nonzero((svi_emb == 0).all(dim=1))
        non_zero_rows = torch.nonzero((svi_emb != 0).any(dim=1))
        nzr_mean = svi_emb[non_zero_rows].mean(dim=0)
        svi_emb[zero_rows] = nzr_mean
        self.svi_emb = svi_emb
        
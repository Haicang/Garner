import os
import time
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

from ..constant import *


__all__ = [
    "save_graph_shapefile_directional",
    "download_osm_data", 
    "check_roadnet_exist",
    "check_road_overlap",
    "generate_road_feat_dict",
    "bag_of_words_encode",
    "check_multiple_str_tags",
    "highway_mapping",
    "binary_repr_of_numeric",
    "encode_sg_road_feat",
    "index_mapping",
    "osm_map_preprocess",
]


##################
## Data loading ##
##################

def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    """This function is copied from https://github.com/cyang-kth/osm_mapmatching .
    """
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = np.arange(0, gdf_edges.shape[0], dtype='int')
    # save the nodes and edges as separate ESRI shapefiles
    print("edge num {}, node num {}".format(len(gdf_edges), len(gdf_nodes)))
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)


def download_osm_data(map_path: str, place: str = 'singapore'):
    assert place in ['singapore', 'nyc']
    place_dict = {'singapore': "Singapore, Singapore", 'nyc': 'New York City, New York, USA'}
    try:
        G = ox.graph_from_place(place_dict[place], network_type="drive", which_result=2)
    except:
        G = ox.graph_from_place(place_dict[place], network_type="drive")
    save_graph_shapefile_directional(G, map_path)


def check_roadnet_exist(dataname):
    """
    Check if the road network data has been downloaded.
    """
    assert dataname in ["singapore", "nyc"]
    map_path = os.path.join(DATA_DIR, dataname, "map")
    return os.path.exists(map_path)


#########################
## Data pre-processing ##
#########################

def check_road_overlap(roads):
    """

    Parameters
    ----------
    roads : pd.Series
        roads is a column from a pandas DataFrame, 
        where each element is a list or an integer of OSM object id.
        This function is to check if there is any overlap between the roads.
    
    """
    osmids = []
    for rid in roads:
        rid = json.loads(rid)
        if isinstance(rid, int):
            rid = [rid]
        osmids.extend(rid)
    osmids = np.array(osmids)
    unique_osmids = np.unique(osmids)
    flag = (len(osmids) == len(unique_osmids))  # True if no overlap
    print('No overlap' if flag else 'Overlap detected')
    return flag


def generate_road_feat_dict(road_feat_col):
    """
    Parameters
    ----------
    road_feat_col : pd.Series
        road_feat_col is a column from a pandas DataFrame, 
        where each element is a str or a list of road features.
        This function is to generate a dictionary that maps each road feature to an integer.
        There could be `nan` in the road_feat_col.
    
    Returns
    -------
    feat_dict : dict
        A dictionary that maps each road feature to an integer.
        If the values are [0, ..., n-1], there is no `nan`;
        if the values are [1, ..., n], there is `nan`.
    """
    unique_tags = []
    nan_flag = False
    for l in road_feat_col.unique():
        if isinstance(l, str):
            if not l.startswith('['):
                unique_tags.append(l)
        elif isinstance(l, float):
            # l is `nan`
            if np.isnan(l):
                nan_flag = True
            else:
                raise NotImplementedError
        else:
            print(l)
            raise NotImplementedError
    unique_tags.sort()

    values = np.arange(0, len(unique_tags))
    if nan_flag:
        values += 1
    feat_dict = dict(zip(unique_tags, values))
    return feat_dict


def bag_of_words_encode(road_feat_col, feat_dict=None):
    """
    Parameters
    ----------
    road_feat_col : pd.Series
        road_feat_col is a column from a pandas DataFrame, 
        where each element is a list of road features.
        This function is to encode the road features into a bag of words.
    feat_dict : dict
        A dictionary that maps each road feature to an integer.
        If the values are [0, ..., n-1], there is no `nan`;
        if the values are [1, ..., n], there is `nan`.
        If `None`, generate a new dictionary.
        But a predefined dictionary is recommended.
    """
    if feat_dict is None:
        feat_dict = generate_road_feat_dict(road_feat_col)
    has_nan = False
    if min(feat_dict.values()) > 0:
        has_nan = True
    num_feat = max(feat_dict.values()) + 1
    encoding = np.zeros((len(road_feat_col), num_feat))

    for i, l in enumerate(road_feat_col):
        if l in feat_dict.keys():
            encoding[i, feat_dict[l]] = 1
        elif isinstance(l, str) and l.startswith('['):
            # string representation of a list
            try:
                l = l.replace("'", '"')
                l = json.loads(l)
                for feat in l:
                    if feat not in feat_dict.keys() and isinstance(feat, str):
                        feat = json.loads(feat)
                    encoding[i, feat_dict[feat]] = 1
            except:
                print(l)
                for i in l:
                    print(type(i), end=',')
                return
        elif isinstance(l, float) and np.isnan(l):
            # l is `nan`
            assert has_nan
            encoding[i, 0] = 1
        else:
            print(l)
            raise TypeError("The type of elements in road_feat_col are not supported.")
    return encoding


def check_multiple_str_tags(roads, col):
    flag = False
    for i, road in roads.iterrows():
        if road['highway'].startswith('['):
            print(i, road['highway'], road['osmid'])
            flag = True
    return flag


def highway_mapping(roads):
    """
    This function is to merge tags with few samples to tags with many samples.
    According to my exploration on roads with multiple `highway` tags, the first tag is usually the main tag.
    """
    for i in roads.index:
        if roads.loc[i, 'highway'].startswith('['):
            roads.loc[i, 'highway'] = roads.loc[i]['highway'].split(',')[0][2:-1]
    
    return roads


def binary_repr_of_numeric(values: pd.Series):
    """
    Parameters
    ----------
    values : pd.Series
        A column from a pandas DataFrame, where each element is a numeric value.
        This function is to encode the numeric values into binary representation.
    """
    max_value = values.max()
    num_bits = int(np.ceil(np.log2(max_value)))
    binary_repr = []
    for v in values:
        brepr = np.binary_repr(int(v), width=num_bits)
        brepr = [int(d) for d in brepr]
        binary_repr.append(brepr)
    binary_repr = np.array(binary_repr)
    return binary_repr
    

def encode_sg_road_feat(edges: gpd.GeoDataFrame) -> np.ndarray:
    """This code is from my jupyter notebook. I generally follow onehot and bag-of-words encoding.
    - `key`: **drop**
    - `osmid`: **drop**
    - `index`: **drop**; `pd.RangeIndex(start=0, stop=45518, step=1)`
    - `oneway`: onehot encoding
    - `lanes`: there are some rows with `nan`. The values are `[nan, 1, ..., 7]`. We encode `nan` as `0` and `'1'` as `1` etc.
    - `ref`: **drop**; some special text
    - `name`: **drop**; However, it tells some relationships
    - `highway`: 
    - `maxspeed`:
    - `reversed`:
    - `length`: binary representation of `int` as bag-of-words vectors
    - `bridge`:
    - `tunnel`:
    - `junction`:
    - `access`:
    - `width`: **drop**
    - `fid`: same as index

    Parameters
    ----------
    edges : gpd.GeoDataFrame
        The roads of the road network.
        
    Returns
    -------
    feat_encoding : np.ndarray
        The encoded road features.
    """
    edges = deepcopy(edges)

    # Encode `oneway`
    oneway_encoding = bag_of_words_encode(edges['oneway'], {0: 0, 1: 1})

    # Encode `lanes`
    lanes = edges['lanes']
    lanes_feat_dict = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8}
    lanes_encoding = bag_of_words_encode(lanes, lanes_feat_dict)

    # Encode `highway`
    # Merge several tags in one road to one tag. Choose the first tag as the main tag.
    edges = highway_mapping(edges)
    highway_feat_dict = {
        'motorway': 0,
        'motorway_link': 0,
        'trunk': 0,
        'trunk_link': 0,
        
        'primary': 1,
        'primary_link': 1,
        
        'secondary': 2,
        'secondary_link': 2,
        
        'tertiary': 3,
        'tertiary_link': 3, 
        'unclassified': 3,
        
        'residential': 4,
        'living_street': 4,
        
        'service': 5,
        'road': 5
    }
    highway_encoding = bag_of_words_encode(edges['highway'], highway_feat_dict)

    # Encode `maxspeed`
    maxspeed_feat_dict = {'10': 1, '15': 2, '20': 3, '25': 4, '30': 5, '40': 6, 
                          '50': 7, '60': 8,  '70': 9, '80': 10, '90': 11}
    maxspeed_encoding = bag_of_words_encode(edges['maxspeed'], maxspeed_feat_dict)

    # Encode `reversed`
    reversed_feat = edges['reversed']
    reversed_feat[reversed_feat == '[False, True]'] = '["False", "True"]'
    reversed_feat_dict = {'False': 0, 'True': 1}
    reversed_encoding = bag_of_words_encode(reversed_feat, reversed_feat_dict)

    # Encode `length`
    length_feat = edges['length']
    length_encoding = binary_repr_of_numeric(length_feat)

    # Encode `bridge`
    bridge_feat = edges['bridge']
    bridge_feat.value_counts()
    bridge_feat_dict = {'yes': 1, 'viaduct': 2}
    bridge_encoding = bag_of_words_encode(bridge_feat, bridge_feat_dict)

    # Encode `tunnel`
    tunnel_feat = edges['tunnel']
    tunnel_feat[tunnel_feat == 'no'] = float("nan")
    tunnel_feat.value_counts()
    tunnel_feat_dict = {'yes': 1, 'building_passage': 2}
    tunnel_encoding = bag_of_words_encode(tunnel_feat, tunnel_feat_dict)

    # Encode `junction`
    junction_feat = edges['junction']
    junction_feat_dict = {'roundabout': 1}
    junction_encoding = bag_of_words_encode(junction_feat, junction_feat_dict)

    # Encode `access`
    access_feat = edges['access']
    access_feat[access_feat == 'restricted'] = float("nan")
    access_feat[access_feat == 'residents'] = float("nan")
    access_feat_dict = {'yes': 1, 'no': 2, 'permissive': 3, 'destination': 4}
    access_encoding = bag_of_words_encode(access_feat, access_feat_dict)

    feat_encoding = np.concatenate([oneway_encoding, lanes_encoding, highway_encoding, 
                                    maxspeed_encoding, reversed_encoding, length_encoding, 
                                    bridge_encoding, tunnel_encoding, junction_encoding, 
                                    access_encoding], axis=1)
    return feat_encoding


def index_mapping(nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame):
    """Map the index of nodes and edges to [0, ..., n-1].
    Remove node and road features, and just save the graph structure and geometry.

    Parameters
    ----------
    nodes : gpd.GeoDataFrame
        The nodes of the road network.
    edges : gpd.GeoDataFrame
        The roads of the road network.

    Returns
    -------
    reindexed_nodes : gpd.GeoDataFrame
        The reindexed nodes.
    reindexed_edges : gpd.GeoDataFrame
        The reindexed edges.
    """
    idx2nodeid = nodes['osmid'].values
    nodeid2idx = dict(zip(idx2nodeid, nodes['osmid'].index.values))
    reindexed_nodes = gpd.GeoDataFrame({'id': nodes.index.values, 'osmid': nodes.osmid, 
                                        'x': nodes.x, 'y': nodes.y, 'geometry': nodes.geometry}, 
                                        nodes.index, geometry='geometry')
    reindexed_nodes = reindexed_nodes.set_index(['id'])

    reindexed_edges = deepcopy(edges[['u', 'v', 'key', 'osmid', 'geometry']])
    reindexed_edges.reset_index(inplace=True)
    reindexed_edges = deepcopy(reindexed_edges[['u', 'v', 'key', 'osmid', 'geometry']])
    for i in range(reindexed_edges.shape[0]):
        reindexed_edges.loc[i, 'u'] = nodeid2idx[reindexed_edges.loc[i, 'u']]
        reindexed_edges.loc[i, 'v'] = nodeid2idx[reindexed_edges.loc[i, 'v']]
    reindexed_edges = reindexed_edges.set_index(['u', 'v', 'key'])
    return reindexed_nodes, reindexed_edges


def osm_map_preprocess(data_dir, dataname='singapore'):
    """
    Parameters
    ----------
    data_dir : str
        Path to the road network shapefile. Default is f"PROJECT_DIR/data/".
    dataname : str
        The name of the dataset. Choose from ['singapore', 'nyc'].
    """
    assert dataname in ['singapore', 'nyc']
    map_path = os.path.join(data_dir, dataname, 'map')

    try:
        nodes = gpd.read_file(os.path.join(map_path, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(map_path, 'edges.shp'))
        print('Use local map data.')
    except:
        if not check_roadnet_exist(dataname):
            os.makedirs(map_path)
        print('Download map data from OSM.')
        download_osm_data(map_path, dataname)
        nodes = gpd.read_file(os.path.join(map_path, 'nodes.shp'))
        edges = gpd.read_file(os.path.join(map_path, 'edges.shp'))
    
    # Remove duplicated edges (some multi edges will also be removed)
    edges = edges.loc[edges['key']==0]

    data_processed_path = os.path.join(data_dir, dataname, 'data_processed')
    if not os.path.exists(data_processed_path):
        os.makedirs(data_processed_path)
    road_feat_path = os.path.join(data_processed_path, 'road_feat.npz')
    if dataname == 'singapore':
        feat_encoding = encode_sg_road_feat(edges)
    elif dataname == 'nyc':
        raise NotImplementedError
    else:
        raise NotImplementedError
    np.savez(road_feat_path, feat_encoding)

    nodes, edges = index_mapping(nodes, edges)
    nodes.to_file(os.path.join(data_processed_path, 'nodes.geojson'), driver='GeoJSON')
    edges.to_file(os.path.join(data_processed_path, 'edges.geojson'), driver='GeoJSON')
    return nodes, edges


###########################
# Preprocessing for tasks #
###########################

def match_speed_to_road(roads: gpd.GeoDataFrame, speed: pd.DataFrame):
    speed.columns = ['speed']
    road_id = []
    speed_values = []
    counts = []
    for i, road in roads.iterrows():
        osm_id = road['osmid']
        if not osm_id.startswith('['):
            osm_id = int(osm_id)
            if osm_id in speed.index:
                road_id.append(i)
                speed_values.append([speed.loc[osm_id, 'speed']])
                counts.append(1)
        else:
            osm_id_lst = osm_id[2:-2].split(', ')
            speed_value_lst = []
            for osm_id in osm_id_lst:
                osm_id = int(osm_id)
                if osm_id in speed.index:
                    speed_value_lst.append(speed.loc[osm_id, 'speed'])
            if len(speed_value_lst) > 0:
                road_id.append(i)
                speed_values.append(speed_value_lst)
                counts.append(len(speed_value_lst))
    road_id = np.array(road_id)
    speed_values = [sum(s) / len(s) for s in speed_values]
    speed_values = np.array(speed_values)
    counts = np.array(counts)
    df = pd.DataFrame({'road_id': road_id, 'speed': speed_values, 'count': counts})
    df = df.set_index('road_id')
    return df

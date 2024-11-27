from copy import deepcopy
import os
import os.path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import torch

from PIL import Image


__all__ = [
    'open_one_image',
    'show_images_of_oid',
    'list_corrupted_images', 
    'remove_corrupted_images', 
    'generate_imagefolder_metadata',
    'svi_pos_to_gdf',
    'align_images_to_roads_nearest',
    'align_images_to_roads_distance',
    'align_images_to_roads_distance_2',
    'align_images_to_roads_wrapper'
]


def open_one_image(oid, angle=0, data_dir='../data/singapore/SVI'):
    angle_dict = {0: 'North', 90: 'East', 180: 'South', 270: 'West'}
    image_path = os.path.join(data_dir, angle_dict[angle], f'{oid}_{angle}.png')
    im = Image.open(image_path)
    return im


def show_images_of_oid(oid, data_dir='../data/singapore/SVI'):
    ims = []
    for angle in [0, 90, 180, 270]:
        ims.append(open_one_image(oid, angle, data_dir))
    return ims


def list_corrupted_images(data_dir='~/'):
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    corrupted_images = []
    for subfolder in subfolders:
        for file in os.listdir(subfolder):
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(subfolder, file)
                # image_list.append(image_path)
                try:
                    im = Image.open(image_path)
                except:
                    corrupted_images.append(image_path)
    return corrupted_images


def remove_corrupted_images(corrupted_images):
    for image in corrupted_images:
        os.remove(image)
    return corrupted_images


def generate_imagefolder_metadata(data_dir):
    """
    Generate metadata for imagefolder dataset.
    `load_dataset` function in `datasets` package can read the filename and parse the objectid and angle of each SVI.
    """
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subfolder in subfolders:
        metadata_file = os.path.join(subfolder, 'metadata.csv')
        assert not os.path.exists(metadata_file), f'{metadata_file} already exists!'
        with open(metadata_file, 'w') as f:
            f.write('file_name,objectid,angle\n')
            for file in os.listdir(subfolder):
                if file.endswith(".jpg") or file.endswith(".png"):
                    objectid, angle = file.split('_')
                    angle = angle.split('.')[0]
                    f.write(f'{file},{objectid},{angle}\n')


def svi_pos_to_gdf(svi_pos: pd.DataFrame, objectid='OBJECTID_1') -> gpd.GeoDataFrame:
    """
    Convert the position of SVI to a GeoDataFrame.
    """
    df = svi_pos
    df_geom = [Point(pos) for pos in df[['POINT_X', 'POINT_Y']].values]
    gdf = gpd.GeoDataFrame(df[objectid], geometry=df_geom, crs="EPSG:4326")
    return gdf


def align_images_to_roads_nearest(roads: gpd.GeoDataFrame, svi_pos_gdf: gpd.GeoDataFrame, objectid='OBJECTID_1', strict=True) -> pd.DataFrame:
    """Use `gpd.sjoin_nearest` to align SVI images to roads.
    Remember to transform the CRS of `roads` and `svi_pos_gdf` to meters, according to 
    https://stackoverflow.com/questions/72073417/userwarning-geometry-is-in-a-geographic-crs-results-from-buffer-are-likely-i

    Parameters
    ----------
    roads : gpd.GeoDataFrame
        Roads in GeoDataFrame format.
    svi_pos_gdf : gpd.GeoDataFrame
        SVI positions in GeoDataFrame format.
    strict : bool, optional
        If True, only keep the closest SVI image to each road, by default True.
    """
    # Transform CRS to meters.
    roads_m = roads.to_crs(crs=3857)
    svi_pos_gdf_m = svi_pos_gdf.to_crs(crs=3857)

    if strict:
        # According to Google API, set `max_distance=50`.
        # Google SVI API doc: https://developers.google.com/maps/documentation/streetview/request-streetview?hl=zh-cn 
        nearest_roads_all = gpd.sjoin_nearest(svi_pos_gdf_m, roads_m, max_distance=50, distance_col='distance')
        # If one SVI is close to multiple roads, keep the one with the smallest distance.
        nearest_roads = nearest_roads_all.sort_values('distance').drop_duplicates(objectid).sort_values(objectid)
    else:
        # Choose multiple matches.
        nearest_roads = gpd.sjoin_nearest(svi_pos_gdf_m, roads_m, max_distance=100, distance_col='distance')
    svi_2_roads = nearest_roads[[objectid, 'index_right']]
    svi_2_roads.columns = ['objectid', 'road_id']
    return svi_2_roads


def align_images_to_roads_distance(roads: gpd.GeoDataFrame, svi_pos_gdf: gpd.GeoDataFrame, threshold=20.,
                                   objectid_col='OBJECTID_1', n_jobs=1) -> pd.DataFrame:
    """
    Align SVI images to roads.
    Remember to transform the CRS of `roads` and `svi_pos_gdf` to meters, according to
    https://stackoverflow.com/questions/72073417/userwarning-geometry-is-in-a-geographic-crs-results-from-buffer-are-likely-i
    """
    # Transform CRS to meters.
    left = deepcopy(svi_pos_gdf[[objectid_col, 'geometry']]).to_crs(crs=3857)
    right = deepcopy(roads[['index', 'geometry']]).to_crs(crs=3857)
    objectid = []
    road_id = []
    distance = []
    if n_jobs == 1:
        # Single process.
        for i in range(left.shape[0]):
            for j in range(right.shape[0]):
                dist = left.iloc[i]['geometry'].distance(right.iloc[j]['geometry'])
                if dist < threshold:
                    objectid.append(left.iloc[i][objectid_col])
                    road_id.append(right.iloc[j]['index'])
                    distance.append(dist)
    else:
        # Parallel processing.
        from joblib import Parallel, delayed
        def _align_one_row(i):
            objectid = []
            road_id = []
            distance = []
            for j in range(right.shape[0]):
                dist = left.iloc[i]['geometry'].distance(right.iloc[j]['geometry'])
                if dist < threshold:
                    objectid.append(left.iloc[i][objectid_col])
                    road_id.append(right.iloc[j]['index'])
                    distance.append(dist)
            return objectid, road_id, distance
        results = Parallel(n_jobs=n_jobs)(delayed(_align_one_row)(i) for i in range(left.shape[0]))
        for result in results:
            objectid.extend(result[0])
            road_id.extend(result[1])
            distance.extend(result[2])
    svi_2_road = pd.DataFrame({'objectid': objectid, 'road_id': road_id, 'distance': distance})
    return svi_2_road


def align_images_to_roads_distance_2(roads: gpd.GeoDataFrame, svi_pos_gdf: gpd.GeoDataFrame, threshold=20.,
                                      objectid_col='OBJECTID_1') -> pd.DataFrame:
    left: gpd.GeoDataFrame = deepcopy(svi_pos_gdf[[objectid_col, 'geometry']]).to_crs(crs=3857)
    right: gpd.GeoDataFrame = deepcopy(roads[['index', 'geometry']]).to_crs(crs=3857)
    right.columns = ['road_id', 'geometry']
    right_buffer = right.buffer(threshold)
    right_buffer_gdf = gpd.GeoDataFrame(data=right[['road_id']], geometry=right_buffer.values, crs='EPSG:3857')
    svi_2_road: gpd.GeoDataFrame = gpd.sjoin(left, right_buffer_gdf)  # The columns are [objectid_col, 'geometry', 'index_right', 'road_id']
    svi_2_road['index_left'] = svi_2_road.index
    # reset a new index for svi_2_road
    svi_2_road.reset_index(inplace=True, drop=True)

    dist = []
    for i, row in svi_2_road.iterrows():
        li = row['index_left']
        ri = row['index_right']
        dist.append(left.loc[li, 'geometry'].distance(right.loc[ri, 'geometry']))
    svi_2_road['distance'] = dist
    svi_2_road.rename(columns={objectid_col: 'objectid'}, inplace=True)
    return svi_2_road[['objectid', 'road_id', 'distance']]


def percent_road_with_svi(roads: gpd.GeoDataFrame, svi_2_roads: pd.DataFrame):
    return np.unique(svi_2_roads['road_id'].values).shape[0] / roads.shape[0]


def count_unique_nonnan(arr, num):
    """
    Count the number of unique non-nan values in an array.
    """
    return np.unique(arr[~np.isnan(arr)]).shape[0] / num


def align_images_to_roads_wrapper(road_path: str, svi_pos_path: str) -> pd.DataFrame:
    """
    Wrapper function for `align_images_to_roads`.
    """
    roads = gpd.read_file(road_path)
    svi_pos = pd.read_csv(svi_pos_path)
    svi_pos_gdf = svi_pos_to_gdf(svi_pos)
    return align_images_to_roads_nearest(roads, svi_pos_gdf)


def count_tensor_nonzero_rows(tensor):
    """
    Count the number of non-zero rows in a tensor.
    """
    return torch.count_nonzero(torch.sum(tensor ** 2, dim=1)).item()


def find_zero_rows(tensor):
    """
    Find the indices of zero rows in a tensor.
    """
    return torch.where(torch.sum(tensor ** 2, dim=1) == 0)[0].tolist()


def main():
    image_dir = os.path.expanduser('~')
    corrupted_images = list_corrupted_images('')
    print('Number of corrupted images: ', len(corrupted_images))
    print('Removing corrupted images...')
    remove_corrupted_images(corrupted_images)
    print('Done!')
    
    
if __name__ == '__main__':
    main()

"""
This file is to generate embedding for SVI.
"""
import os
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import torch
from datasets import load_dataset
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from PIL import Image
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from utils import *


__all__ = [
    'generate_imagefolder_metadata',
    'SVIEncoder',
    'Road_SVI_Embedding',
]


def generate_imagefolder_metadata(data_dir):
    """
    Generate metadata for imagefolder dataset.
    `load_dataset` function in `datasets` package can read the filename and parse the objectid and angle of each SVI.
    """
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    for subfolder in subfolders:
        metadata_file = os.path.join(subfolder, 'metadata.csv')
        assert not os.path.exists(
            metadata_file), f'{metadata_file} already exists!'
        with open(metadata_file, 'w') as f:
            f.write('file_name,objectid,angle\n')
            for file in os.listdir(subfolder):
                if file.endswith(".jpg") or file.endswith(".png"):
                    objectid, angle = file.split('_')
                    angle = angle.split('.')[0]
                    f.write(f'{file},{objectid},{angle}\n')


class SVIEncoder():
    def __init__(self, gpu=-1):
        self.device = torch.device(f'cuda:{gpu}' if gpu >= 0 else 'cpu')
        self.model_name = "openai/clip-vit-large-patch14"
        # self.model_name = "openai/clip-vit-base-patch32"
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            self.model_name, device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, device=self.device)

    def preprocess_dataset(self, dataset):
        """
        dataset: Huggingface dataset object.
        """
        dataset = dataset.map(lambda x: self.processor(images=x["image"], return_tensors="pt"),
                              batched=True)
        dataset.set_format(type='torch', columns=['pixel_values'])
        return dataset

    def embed_images(self, dataset):
        """
        dataset: Huggingface dataset object.
        """

        inference_dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=1)

        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for batch in inference_dataloader:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)
                outputs = self.model(**batch)
                image_embeds = outputs.image_embeds
                embeddings.append(image_embeds)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.to('cpu')
        return embeddings
    

class Road_SVI_Embedding():
    def __init__(self, im_emb: torch.Tensor, im_metadata: pd.DataFrame, 
                 im_2_road: pd.DataFrame, n_roads: int):
        """
        im_emb: Embeddings of images
        im_metadata: Metadata of images, with columns [`objectid`, `angle`]
        im_2_road: Mapping from images to roads, with columns [`objectid`, `road_id`]

        This class can handle the case when the images (im_metadata) are less than in im_2_road.
        """
        self.im_emb = im_emb
        self.im_metadata = im_metadata
        self.im_2_road = im_2_road
        self.n_roads = n_roads
        self.im_emb_on_road: torch.Tensor = None

    def mean_pooling(self) -> torch.Tensor:
        """
        Mean pooling of embeddings of SVI images on each road.
        """
        im_2_road = self.im_2_road
        im_metadata = self.im_metadata
        im_emb = self.im_emb
        n_roads = self.n_roads
        im_metadata['pt_index'] = im_metadata.index

        im_md = im_metadata.merge(im_2_road, on='objectid', how='left')
        im_emb_on_road = torch.zeros(n_roads, im_emb.shape[1])
        im_count = torch.zeros(n_roads)
        for i, row in im_md.iterrows():
            if not np.isnan(row['road_id']):
                road_id = np.int64(row['road_id'])
                im_emb_on_road[road_id] += im_emb[np.int64(row['pt_index'])]
                im_count[road_id] += 1
        im_count[im_count==0] = 1
        im_emb_on_road /= im_count.resize_(im_count.shape[0], 1)
        self.im_emb_on_road = im_emb_on_road
        return im_emb_on_road
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='../data/singapore')
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--svi-dir', type=str, default='SVI')
    parser.add_argument('--svi-pos', type=str, default='svi_sampling_point.csv')
    parser.add_argument('--distance', type=float, default=20.0)
    args = parser.parse_args()

    data_dir = args.data_dir
    svi_dir = os.path.join(data_dir, args.svi_dir)

    # Remove corrupted images
    corrupted_images = list_corrupted_images(svi_dir)
    print('Number of corrupted images: ', len(corrupted_images))
    print('Removing corrupted images...')
    remove_corrupted_images(corrupted_images)
    print('Done!')
    
    # Load hugging face dataset
    generate_imagefolder_metadata(svi_dir)
    dataset = load_dataset("imagefolder", data_dir=svi_dir, split='train')
    print(dataset)

    # Dump `objectid` and `angle`
    metadata_path = os.path.join(data_dir, 'data_processed', 'im_metadata.csv')
    if os.path.exists(metadata_path):
        print(f'{metadata_path} already exists!')
    else:
        metadata_df = {'objectid': dataset['objectid'], 'angle': dataset['angle']}
        metadata_df = pd.DataFrame(metadata_df)
        metadata_df.to_csv(metadata_path, index=False)

    # Generate embeddings and dump
    svi_enc = SVIEncoder(gpu=args.gpu)
    processed_dataset = svi_enc.preprocess_dataset(dataset)
    embeddings = svi_enc.embed_images(processed_dataset)
    emb_path = os.path.join(data_dir, 'data_processed', 'svi_embeddings.pt')
    torch.save(embeddings, emb_path)
    print(f'SVI embeddings saved to {emb_path}!')

    # Align SVI to roads
    svi_pos = pd.read_csv(os.path.join(data_dir, args.svi_pos))
    roads = gpd.read_file(os.path.join(data_dir, 'data_processed', 'edges.geojson'))
    svi_pos_gdf = svi_pos_to_gdf(svi_pos)
    
    
    svi_2_roads = align_images_to_roads_distance_2(roads, svi_pos_gdf, threshold=args.distance, objectid_col='OBJECTID')
    road_svi_encoder = Road_SVI_Embedding(embeddings, metadata_df, svi_2_roads, roads.shape[0])
    im_emb_on_road = road_svi_encoder.mean_pooling()
    torch.save(im_emb_on_road, os.path.join(data_dir, 'data_processed', 'svi_embeddings_on_roads.pt'))


if __name__ == '__main__':
    main()

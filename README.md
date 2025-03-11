# Garner
[NeurIPS 2024] Road Network Representation Learning with the Third Law of Geography


## Environment Setup

Dependencies:
```
python=3.11.8
pytorch=2.1.2
dgl=2.1.0.cu118
networkx=3.1
numpy=1.26.4
pandas=2.1.1
geopandas=0.14.0
osmnx=1.7.1

transformers=4.33.3
datasets=2.14.6
```

`osmnx` has some dependencies that make it very slow to install with conda. Thus I would recommend using pip to install it and its dependencies. 

You can use the following steps to install the dependencies:
```
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c dglteam/label/cu118 dgl
conda install torchdata pydantic
conda install -c plotly plotly=5.18.0
conda install networkx pandas jupyter notebook jupyterlab nb_conda_kernels geopandas=0.14.0 folium fiona shapely

pip install torchmetrics rasterio osmnx==1.8.1
```


## Data



## Usage

Change the directories in `src/constants.py` to match your system. You may also want to change the data path in `src/data.py`.

Download data from https://drive.google.com/drive/folders/12gaB6FZYEChRcDRCDQSxllEb_lStI3ag?usp=drive_link, put it in the project directory, unzip the file. The unziped `data` folder should be in the project directory. (`data` folder and `src` should be in the same layer.)

Use `python scripts/pretrain.py` to pretrain the model. You need to change the hyper-parameters in the file. You will also need to change the project path in the second line of this file. This script also contains some evaluation code, which is commented at the end of this file. You can comment the training part and uncomment the evaluation part to evaluate the representation.
import os
import sys

import torch
import numpy as np

from .model import *
from .data import RoadDataset
from .task import road_function_prediction, average_speed_prediction
# from .task import road_function_prediction, average_speed_prediction, road_retrieval
from .constant import *
from .utils import check_file_dir_exists


# TODO: write __all__ after finishing the project.


def self_supervised_learning(argv: list = None):
    if argv is None:
        argv = sys.argv[1:]
    model_name = argv[0]
    trainer = choose_trainer(model_name)
    args = trainer.get_args(argv[1:])
    trainer.set_args(args)
    road_dataset = RoadDataset(args.dataset, args.data_dir, match_distance=args.match_distance)
    trainer.set_env(args)
    processed_data = trainer.data_process(road_dataset)
    emb = trainer.train(road_dataset, **processed_data)
    if args.emb_filename is not None:
        check_file_dir_exists(args.emb_filename)
        torch.save(emb, args.emb_filename)
    return emb


def evaluate(argv: list = None):
    if argv is None:
        argv = sys.argv[1:]
    task_name = argv[0]
    model_name = argv[1]
    if task_name in ['road_function', 'average_speed']:
        emb_trainer = choose_trainer(model_name)
        args = emb_trainer.get_args(argv[2:])
        emb = torch.load(args.emb_filename)
        if task_name == 'road_function':
            road_function_prediction(args, emb)
        elif task_name == 'average_speed':
            average_speed_prediction(args, emb)
        else:
            raise ValueError(f'Unknown task name: {task_name}')
    # elif task_name in ['road_retrieval']:
    #     road_retrieval(argv[2:])
    else:
        raise ValueError(f'Unknown task name: {task_name}')
    
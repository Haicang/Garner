"""
This file contains classed for downstream tasks.
"""
import os
import abc
import json
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_auroc, multiclass_f1_score
from torchmetrics.functional.regression import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit, ShuffleSplit

from ..utils import *
from ..constant import *


class LogReg(nn.Module):
    def __init__(self, hid_dim, n_classes):
        super(LogReg, self).__init__()

        self.fc = nn.Linear(hid_dim, n_classes)

    def forward(self, x):
        ret = self.fc(x)
        return ret
    

class EvaluationTrainer():
    def __init__(self, args, times=30, split='stratified_shuffle', test_size=0.2):
        self.args = args
        self.device = choose_device(args.gpu)
        self.times = times
        self.cv = int(1 / test_size)
        self.test_size = test_size
        # No need to add multi-process support, 
        #     since the current version can fully utilize the GPU's CUDA cores.
        # Support for various data splits.
        assert split in ['kfold', 'stratified_shuffle', 'shuffle']
        if split == 'kfold':
            assert times % self.cv == 0
            n_repeats = times // self.cv
            self.splitter = RepeatedStratifiedKFold(n_splits=self.cv, n_repeats=n_repeats)
        elif split == 'stratified_shuffle':
            self.splitter = StratifiedShuffleSplit(n_splits=times, test_size=test_size)
        elif split == 'shuffle':
            self.splitter = ShuffleSplit(n_splits=times, test_size=test_size)

    @abc.abstractmethod
    def fit(self, feat, label):
        pass


class EvaluationMetrics():
    """
    For multi-class classification, the metrics are:
        - accuracy (micro_f1)
        - macro_f1
        - auc
    and `n_classes` is required.
    """
    def __init__(self, mode, **kwargs) -> None:
        # Multi-class classification.
        assert mode in ['classification', 'regression']
        self.funcs = {}
        if mode == 'classification':
            assert 'n_classes' in kwargs.keys()
            n_classes = kwargs['n_classes']
            self.funcs['accuracy'] = partial(multiclass_accuracy, num_classes=n_classes, average='micro')
            self.funcs['macro_f1'] = partial(multiclass_f1_score, num_classes=n_classes, average='macro')
            self.funcs['auroc'] = partial(multiclass_auroc, num_classes=n_classes, average='macro')
        elif mode == 'regression':
            self.funcs['mae'] = mean_absolute_error
            self.funcs['rmse'] = lambda x, y: mse_loss(x, y, reduction='mean').sqrt()
            self.funcs['mape'] = mean_absolute_percentage_error
            self.funcs['mare'] = mean_absolute_relative_error
            self.funcs['r_squared'] = r2_score
        self.values = {}
        for k in self.funcs.keys():
            self.values[k] = []

    def reset_cache(self):
        for k in self.values.keys():
            self.values[k] = []
    
    def compute(self, input, target, update=False):
        """
        Parameters
        ----------
        input : torch.Tensor, shape [n_nodes, n_classes]
        target : torch.Tensor, shape [n_nodes]
        update : bool, default False
            Whether to update the cache.
        """
        with torch.no_grad():
            ret = {}
            for key, func in self.funcs.items():
                ret[key] = func(input, target).cpu().item()
                if update:
                    value = ret[key]
                    if key in ['accuracy', 'macro_f1']:
                        value *= 100.0
                    self.values[key].append(value)
            return ret
        
    def dump_results_json(self):
        results = {}
        for k, v in self.values.items():
            arr_v = np.array(v)
            results[k] = arr_v.mean()
            results[k + '_std'] = arr_v.std()
        return json.dumps(results)


class LogRegTrainer(EvaluationTrainer):
    """
    Used for training simple model like MLP or LogReg.
    Only show the total results.
    For road function prediction, there are 6 classes (i.e., 0-5). 
    There is another class "special use" with label (-100), which will be excluded with `_check_inputs`.
    """
    def __init__(self, args, task_name, times=30, split='shuffle', test_size=0.2):
        super(LogRegTrainer, self).__init__(args, times, split, test_size)
        self.task_name = task_name
        self.metrics: EvaluationMetrics = None

    @staticmethod
    def _check_inputs(feat, label):
        """
        The values of label should be in [0, ..., label.max()], negative values will be removed.
        """
        assert feat.shape[0] == label.shape[0]
        idx = label >= 0
        return feat[idx], label[idx]
    
    def fit(self, feat, label):
        """
        feat: torch.Tensor, shape [n_nodes, hid_dim]
        label: torch.Tensor, shape [n_nodes]
        """
        feat, label = self._check_inputs(feat, label)
        args = self.args
        device = self.device
        times = self.times
        n_classes = torch.unique(label).shape[0]
        self.metrics = EvaluationMetrics('classification', n_classes=n_classes)

        feat = feat.to(device)
        label = label.to(device)

        splitter = self.splitter
        for i, (train_index, test_index) in enumerate(splitter.split(feat.cpu(), label.cpu())):
            model = LogReg(feat.shape[1], n_classes).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr2, 
                                    weight_decay=args.wd2)
            criterion = nn.CrossEntropyLoss()

            model.train()
            for epoch in range(500):
                opt.zero_grad()
                logits = model(feat[train_index])
                loss = criterion(logits, label[train_index])
                loss.backward()
                opt.step()

            with torch.no_grad():
                model.eval()
                logits = model(feat[test_index])
                probs = torch.softmax(logits, dim=1)
                # Compute metrics.
                self.metrics.compute(probs, label[test_index], update=True)

        print(f'''{self.task_name}: {times} runs | {self.metrics.dump_results_json()}''')
        return self.metrics.values


class LinRegTrainer(EvaluationTrainer):
    """
    Used for training simple model like MLP or Linear Regression.
    """
    def __init__(self, args, task_name, times=30, split='shuffle', test_size=0.2):
        super().__init__(args, times, split, test_size)
        self.task_name = task_name
        self.metrics: EvaluationMetrics = None

    def fit(self, feat, target):
        """
        feat: torch.Tensor, shape [n_nodes, hid_dim]
        target: torch.Tensor, shape [n_nodes]
        """
        args = self.args
        device = self.device
        times = self.times
        self.metrics = EvaluationMetrics('regression')

        feat = feat.to(device)
        target = target.to(device)

        splitter = self.splitter
        for i, (train_index, test_index) in enumerate(splitter.split(feat.cpu(), target.cpu())):
            model = nn.Linear(feat.shape[1], 1).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr2, 
                                    weight_decay=args.wd2)
            criterion = nn.MSELoss()

            model.train()
            for epoch in range(1000):
                opt.zero_grad()
                logits = model(feat[train_index]).squeeze()
                loss = criterion(logits, target[train_index])
                loss.backward()
                opt.step()

            with torch.no_grad():
                model.eval()
                logits = model(feat[test_index]).squeeze()
                # Compute metrics.
                self.metrics.compute(logits, target[test_index], update=True)

        print(f'''{self.task_name}: {times} runs | {self.metrics.dump_results_json()}''')
        return self.metrics.values
    

def load_average_speed_data(dataname: str):
    data_dir_dict = {
        'singapore': os.path.join(PROJECT_DIR, 'data/singapore/task/average_speed.csv'),
        'nyc': os.path.join(PROJECT_DIR, 'data/nyc/task/average_speed.csv'),
    }
    assert dataname in data_dir_dict.keys()
    if dataname == 'singapore':
        speed_df = pd.read_csv(data_dir_dict[dataname])
        speed_df = speed_df.loc[speed_df['speed'] > 0]
        idx = torch.from_numpy(speed_df['road_id'].values).long()
        speed = torch.from_numpy(speed_df['speed'].values).float()
        return idx, speed
    else:
        speed_df = pd.read_csv(data_dir_dict[dataname])
        speed_df = speed_df.loc[speed_df['speed'] > 0]
        idx = torch.from_numpy(speed_df['road_id'].values).long()
        speed = torch.from_numpy(speed_df['speed'].values).float()
        return idx, speed


def road_function_prediction(args, emb):
    eval_trainer = LogRegTrainer(args, 'Road Function Classification', test_size=args.test_ratio)
    data_dir = os.path.join(DATA_DIR, args.dataset)
    label = np.load(os.path.join(data_dir, 'task/road_function.npz'))['arr_0']
    label = torch.from_numpy(label).long()
    eval_trainer.fit(emb, label)
    

def average_speed_prediction(args, emb):
    eval_trainer = LinRegTrainer(args, 'Average Speed Prediction', split='shuffle', test_size=args.test_ratio)
    idx, speed = load_average_speed_data(args.dataset)
    eval_trainer.fit(emb[idx], speed)

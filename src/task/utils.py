"""
This file stores the utility functions for downstream tasks.
"""

import os

import numpy as np
import pandas as pd


def compute_trajectory_time(trjs: pd.DataFrame, sort=False) -> pd.DataFrame:
    """
    Compute the time for each trajectory.

    Args:
        trjs: DataFrame, trajectory data, each row is a GPS point on a trajectory.
        sort: bool, whether to sort the trajectory data by `trj_id` and `pingtimestamp`.
    """
    if sort:
        trjs = trjs.sort_values(by=['trj_id', 'pingtimestamp'])
    grouped = trjs.groupby('trj_id')['pingtimestamp']
    time = grouped.max() - grouped.min()
    time = time.to_frame()
    return time

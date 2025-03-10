import os
import sys


__all__ = [
    'check_file_dir_exists',
]


def check_file_dir_exists(filepath: str):
    assert '/' in filepath
    dirpath = '/'.join(filepath.split('/')[:-1])
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

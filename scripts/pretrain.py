import sys
sys.path.append('/home_nfs/haicang/workspace/Garner/')
from src import *
from _utils import *


def run_task_v3(dataset, task, model, gpu, logname, emb_file_prefix, runs=list(range(10))):
    argv = [task, model,
            '--gpu', gpu, 
            '--dataset', dataset,
            '--test-ratio', 0.2,
    ]
    argv = [str(arg) for arg in argv]
    check_file_dir_exists(logname)
    with open(logname, 'w') as f:
        sys.stdout = f
        for idx in runs:
            new_argv = argv + [
                '--emb-filename', emb_file_prefix + f'_{idx}.pt',
                ]
            new_argv = [str(arg) for arg in new_argv]
            evaluate(new_argv)
            sys.stdout.flush()


GPU=3
DATASET = 'singapore'
# DATASET = 'nyc'
INDEX=0

# Parameter setting for training and evaluation.
dataset = DATASET
model = 'garner'
model_v = model
argv = [model, 
        '--gpu', GPU, 
        '--dataset', DATASET,
        '--sim-neg-d', 22,
]

# Training script
logname = f'logs_pretrain/{dataset}_{model_v}_{INDEX}.txt'
check_file_dir_exists(logname)
with open(logname, 'w') as f:
    sys.stdout = f
    # Multiple runs
    for idx in range(10):
        new_argv = argv + [
            '--epochs', 2500,
            '--emb-filename', f'savings/{dataset}_{model_v}_emb_{idx}.pt',
            ]
        new_argv = [str(arg) for arg in new_argv]
        self_supervised_learning(new_argv)
        sys.stdout.flush()

# # Evaluation script
# for task in ['road_function', 'average_speed']:
#     logname = f'logs_eval/{dataset}_{task}_{model_v}_{INDEX}.txt'
#     emb_file_prefix = f'savings/{dataset}_{model_v}_emb'
#     run_task_v3(dataset, task, model, GPU, logname, emb_file_prefix)

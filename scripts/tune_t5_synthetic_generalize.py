from library.t5_tuning import *
from rich import print
from functools import partial
import torch

def process_synthetic(batch, prompt_type='Default'):
    return process(batch['description'], batch['code'], prompt_type=prompt_type)

def make_dataset(max_depth=5, min_depth=1, **kwargs):
    data_files = [f'synthetic_depth_{d}.csv' for d in range(min_depth, max_depth + 1)]
    return load_dataset('data/', data_files=data_files, **kwargs)

def run():
    print('CUDA', torch.cuda.is_available())
    prompt_type = 'Least-to-Most'
    # Generalize from 1-4 to length 5
    restricted = make_dataset(min_depth=1, max_depth=4)
    generalize = make_dataset(min_depth=5, max_depth=5)
    tune_model(restricted, partial(process_synthetic, prompt_type=prompt_type),
               name='synthetic', prefix_dir='/scratch/lsaldyt/experiments/',
               freeze=True)
    test_model(generalize,
               name='synthetic', prefix_dir='/scratch/lsaldyt/experiments/')

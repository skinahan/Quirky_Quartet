from library.t5_tuning import *
from rich import print
from functools import partial
from datasets import Dataset

from .tune_t5_synthetic_generalize import *

def run():
    prompt_type = 'Least-to-Most'
    mixed_dataset = make_dataset(min_depth=1, max_depth=5)
    mixed_dataset = mixed_dataset['train'].train_test_split(test_size=0.2)
    tune_model(mixed_dataset, partial(process_synthetic, prompt_type=prompt_type),
               name='synthetic_all', prefix_dir='/scratch/lsaldyt/experiments/',
               freeze=False, use_gpu=True)
    test_model(mixed_dataset,
               name='synthetic_all', prefix_dir='/scratch/lsaldyt/experiments/')

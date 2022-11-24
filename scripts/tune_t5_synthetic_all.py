from library.t5_tuning import *
from rich import print

def run():
    max_depth = 4
    data_files = [f'synthetic_depth_{d}.csv' for d in range(1, max_depth + 1)]
    dataset = load_dataset('data/', data_files=data_files)
    print(dataset)
    1/0
    tune_model(dataset, preprocess)
    test_model(dataset)

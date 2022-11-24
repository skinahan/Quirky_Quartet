from library.t5_tuning import *
from rich import print

def run():
    # dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    # example = dataset['train'][0]
    # print("Code:", example["code"])
    # print("Docstring:", example["docstring"])
    tune_model(dataset, preprocess)
    test_model(dataset)

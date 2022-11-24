from library.t5_tuning import *
from rich import print

def process_codexglue(examples):
    return process(examples['docstring'], examples['code'], prompt_type='Default')

def run():
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    example = dataset['train'][0]
    print("Code:", example["code"])
    print("Docstring:", example["docstring"])
    tune_model(dataset, process_codexglue, prefix_dir='/scratch/lsaldyt/experiments/')
    test_model(dataset, prefix_dir='/scratch/lsaldyt/experiments/')

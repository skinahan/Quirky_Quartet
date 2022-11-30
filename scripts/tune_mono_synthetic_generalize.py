from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from library.huggingface_tuning import *
from rich import print
from functools import partial
import torch

def process_synthetic(batch, tokenizer=None, prompt_type='Default'):
    return process(batch['description'], batch['code'], tokenizer=tokenizer, 
                   prompt_type=prompt_type)

def make_dataset(max_depth=5, min_depth=1, **kwargs):
    data_files = [f'synthetic_depth_{d}.csv' for d in range(min_depth, max_depth + 1)]
    return load_dataset('data/', data_files=data_files, **kwargs)

def run():
    # model     = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-16B-mono')
    # tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-16B-mono")
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")

    print('CUDA', torch.cuda.is_available())
    prompt_type = 'Least-to-Most'
    # Generalize from 1-4 to length 5
    restricted = make_dataset(min_depth=1, max_depth=4)
    generalize = make_dataset(min_depth=5, max_depth=5)
    tune_model(model, tokenizer, restricted, 
               partial(process_synthetic, tokenizer=tokenizer, prompt_type=prompt_type),
               name='synthetic', prefix_dir='/scratch/lsaldyt/experiments/',
               test_dataset=generalize,
               freeze=False, use_gpu=True)
    # test_model(generalize,
    #            name='synthetic', prefix_dir='/scratch/lsaldyt/experiments/')


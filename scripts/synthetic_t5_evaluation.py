import csv
import ast
from rich import print
from rich.progress import track
from copy import copy
from time import sleep

import contextlib
import signal
import pytorch_lightning as pl

from transformers import RobertaTokenizer, T5ForConditionalGeneration

from library.openai_synthetic_dataset import test_cases
from library.prompts import to_prompt
from library.api_wrapper import *
from library.t5_model import CodeT5

from .synthetic_evaluation import *

def evaluate_t5(model, tokenizer, prompts):
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        # print(input_ids)
        print('input', tokenizer.decode(input_ids[0], skip_special_tokens=False))
        print('inshape', input_ids.shape)
        try:
            outputs = model.generate(input_ids, max_length=128)
        except AttributeError:
            outputs = model.model.generate(input_ids, max_length=128)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
        print('output', decoded)
        print('outshape', outputs.shape)
        yield decoded

def batch_evaluate(prompt_batch, metadata, batch_size, n_samples, model, tokenizer, test_inputs):
    ''' Run T5 in place of Codex '''
    result = list(evaluate_t5(model, tokenizer, prompt_batch))
    print('T5 Evaluation result')
    for code in result:
        print('(start t5 code output)')
        print(code)
        print('(stop t5 code output)')
    print('Result')
    for i, meta in enumerate(metadata):
        extended, problem_id, description, code, md5_hash, *labels = meta
        for k in range(n_samples):
            print('Sample', k)
            codex_out = result[i * n_samples + k]
            behavior, accuracy, outputs = evaluate_output(codex_out, test_inputs, labels)
            print(behavior, accuracy)
            yield ([problem_id, description, md5_hash, extended, k,
                    behavior, accuracy] + outputs)


def run():
    tokenizer  = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    checkpoint = True

    if checkpoint:
        # checkpoint_path = 'results/csv_data/synthetic/version_0/checkpoints/epoch=15-step=13520.ckpt'
        # checkpoint_path = 'results/csv_data/synthetic_all/version_0/checkpoints/epoch=11-step=46176.ckpt'
        checkpoint_path = 'results/synthetic_t5/version_4/checkpoints/epoch=1-step=6756.ckpt'
        print(checkpoint_path)
        model = CodeT5.load_from_checkpoint(checkpoint_path)
    else:
        print('pretrained')
        model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    print(model)

    n_samples  = 1  # Since the problem is already very slow
    batch_size = 20 # The maximum batch size for 500 tokens & 150,000 tokens / minute
    # Also, batch size needs to be divisible by the number of samples
    filtered_test_cases = copy(test_cases)
    # filtered_test_cases.pop('whitespace') # Seems to be unfairly evaluated
    test_inputs = [v for k, v in sorted(filtered_test_cases.items(), key=lambda t:t[0])]
    api_key = read_config()

    headers = (['id', 'description', 'md5', 'extended', 'sample',
                'behavior', 'accuracy']
               + list(sorted(filtered_test_cases.keys())))

    for depth in range(1, 6):
        print(depth)
        data_filename = f'data/synthetic_depth_{depth}.csv'
        n_probs = linecount(data_filename)
        with open(data_filename, 'r') as infile:
            with open(f'results/t5_tuned_synthetic_depth_{depth}.csv', 'w') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                writer.writerow(headers)
                headers = next(reader)
                description = f'Depth {depth} problems'
                for batch, metadata in batch_generator(reader, description, n_probs, n_samples, batch_size):
                    for row in batch_evaluate(batch, metadata, batch_size, n_samples, model, tokenizer, test_inputs):
                        writer.writerow(row)

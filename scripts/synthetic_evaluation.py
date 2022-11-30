import csv
from library.api_wrapper import *
from library.openai_synthetic_dataset import test_cases
import ast
from rich import print
from rich.progress import track
from copy import copy
from time import sleep

import contextlib
import signal

from library.prompts import to_prompt

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def parse_output(output, stop_at_indent_end=False, first_function=False):
    ''' Parse Codex output:
        The first line will be a docstring prompt, then the next line will start
        with a def statement. All lines afterwards should be indented until the
        function ends, and output after this is ignored. '''
    kept = ''
    if stop_at_indent_end:
        indented = False
        for i, line in enumerate(output.splitlines()):
            # print(f'{i:3}: {line}')
            if line.startswith('    '):
                indented = True
                kept += line + '\n'
            else:
                if indented:
                    break # Stop saving input
                else:
                    kept += line + '\n'
    else:
        kept = output
    print('(Parsed code):')
    for line in kept.splitlines():
        print(f'    {line}')
    print('(End of parsed code)')
    node = ast.parse(kept)
    assert isinstance(node, ast.Module)
    for element in node.body:
        if isinstance(element, ast.FunctionDef):
            func_name = element.name
            if first_function:
                break
            # Otherwise uses the last defined function allowing better composition
    else:
        raise SyntaxError(f'No function definition found!')
    obj = compile(node, filename='<ast>', mode='exec')
    namespace = dict()
    exec(obj, namespace)
    return namespace[func_name]

def evaluate_output(codex_output, test_inputs, labels):
    ''' Evaluate Codex on a particular string manipulation problem, using input and
        output pairs. Track success/error type and accuracy '''
    try:
        function = parse_output(codex_output)
        correct = 0
        print('Test outputs:')
        outputs = []
        for inp, lbl in zip(test_inputs, labels):
            with time_limit(5.):
                out = function(inp)
            print(f'    {out == lbl:5}: {out}')
            print(lbl)
            if out == lbl:
                correct += 1
            outputs.append(out)
        accuracy = correct / len(labels)
        behavior = 'Success'
    except Exception as e:
        accuracy = 0
        behavior = type(e).__name__ # Track types of exception
        outputs = [''] * len(labels)
    return behavior, accuracy, outputs

def linecount(filename):
    with open(filename, 'r') as infile:
        return sum(1 for line in infile)

def batch_evaluate(prompt_batch, metadata, batch_size, n_samples, api_key, test_inputs):
    ''' Run codex in batch mode '''
    result = run_codex(api_key, prompt_batch, batch=True) # Will be an ordered list of prompt + response
    print(result)
    for i, meta in enumerate(metadata):
        extended, problem_id, description, code, md5_hash, *labels = meta
        print(code)
        for k in range(n_samples):
            codex_out = result[i * n_samples + k]
            behavior, accuracy, outputs = evaluate_output(codex_out, test_inputs, labels)
            print(behavior, accuracy)
            yield ([problem_id, description, md5_hash, extended, k,
                    behavior, accuracy] + outputs)

def batch_generator(csvreader, description, total, n_samples, batch_size):
    ''' Generator prompt batches and their metadata '''
    metadata = []
    batch    = []
    for row in track(csvreader, description=description, total=total):
        problem_id, description, code, md5_hash, *labels = row
        problem_id = int(problem_id)
        for prompt_type in ['Default', 'Least-to-Most']:
            metadata.append([prompt_type] + list(row))
            prompt = to_prompt(description, prompt_type=prompt_type)
            # print(prompt)
            # print(code)
            # print(labels)
            # print(test_cases)
            samples = [prompt] * n_samples
            batch.extend(samples)
            if len(batch) == batch_size:
                yield batch, metadata
                metadata = []
                batch    = []

def run():
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
        depth = 5
        data_filename = f'data/synthetic_depth_{depth}.csv'
        n_probs = linecount(data_filename)
        with open(data_filename, 'r') as infile:
            with open(f'results/synthetic_depth_{depth}.csv', 'w') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                writer.writerow(headers)
                headers = next(reader)
                description = f'Depth {depth} problems'
                for batch, metadata in batch_generator(reader, description, n_probs, n_samples, batch_size):
                    for row in batch_evaluate(batch, metadata, batch_size, n_samples, api_key, test_inputs):
                        writer.writerow(row)
                    sleep(69) # For good measure

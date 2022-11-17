import csv
from library.api_wrapper import *
from library.openai_synthetic_dataset import test_cases
import ast

def to_prompt(description):
    return f'"""Write a python function to {description}"""\ndef'

def parse_output(output):
    ''' Parse Codex output:
        The first line will be a docstring prompt, then the next line will start
        with a def statement. All lines afterwards should be indented until the
        function ends, and output after this is ignored. '''
    kept = ''
    indented = False
    for i, line in enumerate(output.splitlines()):
        print(f'{i:3}: {line}')
        if line.startswith('    '):
            indented = True
            kept += line + '\n'
        else:
            if indented:
                break # Stop saving input
            else:
                kept += line + '\n'
    print('Kept:')
    print(kept)
    print('(Done)')
    node = ast.parse(kept)
    assert isinstance(node, ast.Module)
    for element in node.body:
        if isinstance(element, ast.FunctionDef):
            func_name = element.name
            break
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
        outputs = []
        for inp, lbl in zip(test_inputs, labels):
            out = function(inp)
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

def run():
    test_inputs = [v for k, v in sorted(test_cases.items(), key=lambda t:t[0])]
    api_key = read_config()

    headers = (['id', 'description', 'md5', 'behavior', 'accuracy']
               + list(sorted(test_cases.keys())))

    samples = 10
    for depth in range(1, 6):
        with open(f'data/synthetic_depth_{depth}.csv', 'r') as infile:
            with open(f'results/synthetic_depth_{depth}.csv', 'w') as outfile:
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                writer.writerow(headers)
                headers = next(reader)
                for row in reader:
                    problem_id, description, md5_hash, *labels = row
                    problem_id = int(problem_id)
                    print(description)
                    prompt = to_prompt(description)
                    codex_out = run_codex(api_key, prompt)
                    total = prompt + codex_out
                    print(total)
                    behavior, accuracy, outputs = evaluate_output(total, test_inputs, labels)
                    print(behavior, accuracy)
                    writer.writerow([problem_id, description, md5_hash, behavior, accuracy]
                                    + outputs)

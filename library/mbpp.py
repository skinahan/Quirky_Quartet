# Module to read MBPP dataset and test prompting techniques
# using codex

from codex_code_gen import *

import re, ast
from datasets import load_dataset

def read_mbpp(sanitized=False):
    """
    Dataset description:
    Has two versions - full (crowd-sourced) and sanitized (verified by authors)
    Split into train, test, validation and prompt
    Each instance has features -
    'task_id'               -   Index for the dataset
                                Split for full dataset (train/test/prompt/validation) - (374, 500, 10, 90)
                                'test'       - Task IDs 11-510 are used for testing. (500 samples)
                                'prompt'     - Task IDs 1-10 were used for few-shot prompting and not for training. (10 samples)
                                'validation' - Task IDs 511-600 were used for validation during fine-tuning. (90 samples)
                                'train'      - Task IDs 601-974 are used for training. (374 samples)
                                Split for sanitized dataset (train/test/prompt/validation) - (120, 257, 7, 43)
    'text'                  -   Problem description
    'code'                  -   Gold solution
    'test_list'             -   Simple tests to evaluate the correctness of the generated code
    'test_setup_code'       -   Gives additional information for the tests (eg: setup tree for testing the code)
    'challenge_test_list'   -   Additional challenging testcases to check for edge cases
    """
    if not sanitized:
        dataset = load_dataset("mbpp")
    else:
        dataset = load_dataset("mbpp", "sanitized")
    return dataset

def visualize_mbpp_instance(instance):
    """
    Given an instance from the dataset (eg: dataset['train'][i]), pretty print the sample
    """
    for k in instance.keys():
        t = type(instance[k])
        print("{0}: ".format(k))
        if t == list or t == tuple:
            for i in instance[k]:
                print(i)
        else:
            print(instance[k])
        print()

def create_codex_query(text, task=""):
    """
    Given question text from MBPP dataset, create query in proper format for codex
    Set task to explicitly ask for python function (default) or cpp code or anything else
    (hopefully ensures properly formatted responses)
    """
    task = task if task != "" else "Write a python function to solve the above question."
    query = "Question:\n" + text + "\n"
    query = query + "Task:\n" + task + "\n"
    query = query + "Answer:"
    return query

def parse_py_from_codex(codex_out):
    """
    Does basic parse to get python output from codex using simple regex match
    (regex assumes some basic structure when matching)
    Validates python output by trying to build an AST
    Note: Not completely robust
    Known issues: Misses python code with (print "something") statements at the end as it is not valid Python3 syntax
    """
    pattern = re.compile(r"(.*)\n\s+(Question|Q|Explanation|Answer|Code|Testcases|Input|\*)", flags=re.S)
    match = re.search(pattern, codex_out)
    code = match.group(1).strip() if match is not None else codex_out.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return (False, codex_out)
    return (True, code)

def execute_py_code(code):
    """
    Executes given python code by building and compiling an AST
    """
    try:
        node = ast.parse(code)
    except SyntaxError:
        print("Unable to parse code")
        return False
    obj = compile(node, filename="<ast>", mode="exec")
    try:
        exec(obj)
    except AssertionError:
        return False
    except Exception as e:
        print("Code fails on execution")
        return False
    return True

# Todo:
# Get function names from AST
# Modify assert statements to use function names and definition as given in output
if __name__ == "__main__":
    api_key = read_config()
    data = read_mbpp(sanitized=False)
    
    print("---- Example from train set ----")
    idx = 0
    split = "train"
    instance = data[split][idx]
    visualize_mbpp_instance(instance)
    print("--------------------------------")

    print("---------- Codex Query ---------")
    query = create_codex_query(instance["text"])
    print(query)
    print("--------------------------------")

    print("---------- Codex Call ----------")
    codex_out = run_codex(api_key, query)
    print(codex_out)
    print("--------------------------------")

    print("---------- Parse Code ----------")
    parse_status, code = parse_py_from_codex(codex_out)
    print("Parse status = {0}".format("Success" if parse_status else "Fail"))
    if parse_status == True:
        print("Code: ")
        print(code)
    print("--------------------------------")

    # Uncomment this if you want to check whether the code executes
    # if parse_status == True:
    #     print("------ Try Code Execution ------")
    #     exec_status = execute_py_code(code)
    #     print("Execution status = {0}".format("Success" if exec_status else "Fail"))
    #     print("--------------------------------")
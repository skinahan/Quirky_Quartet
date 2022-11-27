# Module to read MBPP dataset and test prompting techniques
# using codex

from api_wrapper import *

import os, re, ast, time, json, pickle
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

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

def create_codex_query(text, task="", prompt=""):
    """
    Given question text from MBPP dataset, create query in proper format for codex
    Set task to explicitly ask for python function (default) or cpp code or anything else
    (hopefully ensures properly formatted responses)
    """
    task = task if task != "" else "Write a python function to solve the above question."
    query = (prompt + "\n" if prompt != "" else "") + "Question:\n" + text + "\n"
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
    codex_out = codex_out.strip()
    # pattern = re.compile(r"((\'|\"|Code|Answer)*\n)*(.*)", flags=re.S)
    # match = re.search(pattern, codex_out)
    # code = match.group(3).strip() if match is not None else codex_out.strip()
    pattern = re.compile(r"(.*)\n\s+(Question|Q|Explanation|In|Exercise|Answer|Code|Testcases|Input|\*)", flags=re.S)
    match = re.search(pattern, codex_out)
    code = match.group(1).strip() if match is not None else codex_out.strip()
    try:
        ast.parse(code)
    except SyntaxError:
        return (False, codex_out)
    return (True, code)

def execute_py_code(code, verbose=0):
    """ Executes given python code by building and compiling an AST """
    try:
        node = ast.parse(code)
    except SyntaxError:
        if verbose > 0:
            print("Unable to parse code")
        return False
    try:
        obj = compile(node, filename="<ast>", mode="exec")
        exec(obj)
    except AssertionError:
        if verbose > 0:
            print("Failed on the test assert")
        return False
    except Exception as e:
        if verbose > 0:
            print("Code fails on execution")
        return False
    return True

def mbpp_play_demo(data, api_key, split="train", idx=0):
    print("---- Example from train set ----")
    instance = data[split][idx]
    visualize_mbpp_instance(instance)
    print("--------------------------------")

    print("---------- Codex Query ---------")
    # task = "Write a python function to solve the above question."
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
    print("---------- Done ----------")

def chk_custom_class(data, split):
    """ Print instances in the data if the gold code uses custom class definitions """
    for i in range(data[split].shape[0]):
        code = data[split][i]["code"]
        r = ast.parse(code)
        l = getattr(r, "body")
        for j in l:
            if isinstance(j, ast.ClassDef):
                print(i)

def parse_function_defn(root):
    """ Parse function definitions from the AST """
    fns = []
    args = []
    all = getattr(root, "body")
    func_chk = False
    for i in all:
        if isinstance(i, ast.FunctionDef):
            func_chk = True
            fns.append(i.name)
            args.append([j.arg for j in i.args.args])
    if not func_chk:
        raise Exception("Function is not defined in the AST")
    return fns, args

def get_type_from_ast(node):
    """ Return type of the variable based on AST node """
    if isinstance(node, ast.List):
        return type([])
    elif isinstance(node, ast.Tuple):
        return type(())
    elif isinstance(node, ast.Dict):
        return type({})
    elif isinstance(node, ast.Set):
        return type(set())
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node, ast.UAdd) or isinstance(node, ast.USub):
            return int
        else:
            return bool
    elif isinstance(node, ast.Constant):
        return type(node.value)
    else:
        # Unhandled type
        return type(node)

def get_cmpops_symbols(node):
    if isinstance(node, ast.Eq):
        return "=="
    elif isinstance(node, ast.NotEq):
        return "!="
    elif isinstance(node, ast.Lt):
        return "<"
    elif isinstance(node, ast.LtE):
        return "<="
    elif isinstance(node, ast.Gt):
        return ">"
    elif isinstance(node, ast.GtE):
        return ">="
    elif isinstance(node, ast.Is):
        return "is"
    elif isinstance(node, ast.IsNot):
        return "is not"
    elif isinstance(node, ast.In):
        return "in"
    elif isinstance(node, ast.NotIn):
        return "not in"

def guess_function_argtypes(root):
    """ Guess function argument types from the AST for assert statements """
    types = []
    unk = False
    all = getattr(root, "body")
    for i in all:
        if isinstance(i, ast.Assert):
            for t in i.test.left.args:
                ty = get_type_from_ast(t)
                types.append(ty)
                if ty == type(t):
                    unk = True
    return types, unk

def parse_test_asserts(root):
    """ Parse the assert statements as part of the the tests """
    """ Assumes only single assert statement in the AST """
    """ Assumes only single operator in the comparison """
    """ Assumes only single value to test against in the assert statement """
    params = []
    node = getattr(root, "body")[0]
    if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare):
        for t in node.test.left.args:
            params.append(ast.unparse(t))
        op = node.test.ops[0]
        val = ast.unparse(node.test.comparators[0])
    else:
        print("Unhandled assert statement !!!")
    return params, op, val

def generate_new_asserts(fname, params, op, val):
    """ Generate new assert statements with custom function name for evaluation """
    return "assert " + fname + "(" + ", ".join(params) + ") " + get_cmpops_symbols(op) + " " + val

def eval_codex_out(codex_out, tests, verbose=0):
    """ Parse codex output and perform evaluation if the parse succeeds """
    """ 'tests' is list of params, operator and value for each assert statement in the data  """
    """ Outputs - 0 (tests succeeded) | 1 (tests failed) | 2 (parse error) """
    parse_status, code = parse_py_from_codex(codex_out)
    if parse_status == False:
        if verbose > 0:
            print("Failed at parsing step")
        if verbose > 1:
            print(code)
        return 2, code
    root = ast.parse(code)
    fns, args = parse_function_defn(root)
    success = False
    # Generate tests for each function name
    for i in range(len(fns)):
        fail = False
        # Test each assert statement
        for j in range(len(tests)):
            if fail:
                continue
            params, op, val = tests[j]
            if len(args[i]) != len(params):
                fail = True
            t = generate_new_asserts(fns[i], params, op, val)
            test_code = code + "\n\n" + t
            status = execute_py_code(test_code, verbose=0)  # True if the code runs successfully
            fail = fail or (not status)
            if verbose > 0:
                if status == False:
                    print("Function {0} failed on test {1}".format(i, j))
        if not fail:
            success = True
            break
    return (0 if success else 1), code

def eval_codex(instance, api_key, task="", prompt="", verbose=0):
    """ Run and evaluate codex end-to-end """
    if prompt == "":
        query = create_codex_query(instance["text"], task=task)
    else:
        query = create_codex_query(instance["text"], task=task, prompt=prompt)
    codex_out = run_codex(api_key, query)
    tests = [parse_test_asserts(ast.parse(t)) for t in instance["test_list"]]
    status, code = eval_codex_out(codex_out, tests, verbose=verbose)
    if status == 0:
        if verbose > 0:
            print("Success")
    return status, query, codex_out, code

def batch_eval_0shot(data, api_key, task="", k=1, verbose=0):
    """ Run and evaluate codex on a batch of data in 0-shot setting """
    results = []
    status = []
    for i in range(data.shape[0]):
        print("Example {0}".format(i))
        out = []
        for j in range(k):
            print("Trial {0}".format(j))
            res = eval_codex(data[i], api_key, task=task, prompt="", verbose=verbose)
            out.append(res)
        results.append(out)
        status.append([True if res[0] == 0 else False for res in out])
    return results, status

def create_few_shot_prompt(data, nshot=1, task=""):
    """ Create few shot prompt using prompt split from MBPP dataset """
    nshot = min(nshot, data.shape[0])
    prompt = ""
    for i in range(nshot):
        query = create_codex_query(data[i]["text"], task=task)
        prompt = prompt + query + "\n" + data[i]["code"] + "\n\n"
    return prompt

def batch_eval_fewshot(data, prompt_data, api_key, task="", nshot=1, k=1, verbose=0):
    """ Run and evaluate codex on a batch of data in few-shot setting """
    prompt = create_few_shot_prompt(prompt_data, nshot, task=task)
    results = []
    status = []
    for i in range(data.shape[0]):
        print("Example {0}".format(i))
        out = []
        for j in range(k):
            print("Trial {0}".format(j))
            res = eval_codex(data[i], api_key, task=task, prompt=prompt, verbose=verbose)
            out.append(res)
        results.append(out)
        status.append([True if res[0] == 0 else False for res in out])
    return results, status

def complete_eval_setup(data, split, batch, result_dir, api_key, task="", nshot=1, k=1, verbose=0):
    """ Run and evaluate codex on a batch of data and save results in JSON """
    offset = 0
    batch_size = 100
    mini_batch_size = 1
    result_dir = result_dir + "/" + "{0}-shot".format(nshot)
    os.makedirs(result_dir, exist_ok=True)
    if batch * batch_size > data[split].shape[0]:
        print("Invalid batch")
        return [], []
    batch_data = Dataset.from_dict(data[split][offset + batch * batch_size : min(offset + (batch+1) * batch_size, data[split].shape[0])])
    num_mini_batches = batch_data.shape[0] // mini_batch_size
    for i in range(num_mini_batches):
        mini_batch = Dataset.from_dict(batch_data[i * mini_batch_size : min((i+1) * mini_batch_size, batch_data.shape[0])])
        if nshot > 0:
            results, status = batch_eval_fewshot(mini_batch, data["prompt"], api_key, task=task, nshot=nshot, k=k, verbose=verbose)
        else:
            results, status = batch_eval_0shot(mini_batch, api_key, task=task, k=k, verbose=verbose)
        status = np.any(status, axis=1)
        for idx in range(len(results)):
            query = results[idx][0][1]
            d = {}
            d["query"] = query
            d["success"] = 1 if status[idx] else 0
            for idy in range(k):
                d[idy] = {}
                d[idy]["status"] = results[idx][idy][0]
                d[idy]["codex_out"] = results[idx][idy][2]
                d[idy]["clean_code"] = results[idx][idy][3]
            data_id = offset + batch * batch_size + i * mini_batch_size + idx
            print("Saving {0}".format(data_id))
            save_result_file(d, "{0}/{1}_{2}.json".format(result_dir, split, data_id), is_json=True, is_pickle=False)
        time.sleep(69)

def save_result_file(obj, fname, is_json=True, is_pickle=False):
    if is_json:
        with open(fname, "w") as fp:
            json.dump(obj, fp, indent=4)
    elif is_pickle:
        with open(fname, "wb") as fp:
            pickle.dump(obj, fp)

def load_result_file(fname, is_json=True, is_pickle=False):
    if is_json:
        with open(fname, "r") as fp:
            return json.load(fp)
    elif is_pickle:
        with open(fname, "rb") as fp:
            return pickle.load(fp)

def eval_results(dir, nshot=0, k=1):
    res_dir = "{0}/no_prompt/{1}-shot/".format(dir, nshot)
    fs = ["{0}/{1}".format(res_dir, f) for f in os.listdir(res_dir) if os.path.isfile("{0}/{1}".format(res_dir, f))]
    results = []
    for f in fs:
        data = load_result_file(f, is_json=True)
        results.append([])
        for j in range(k):
            results[-1].append(data[str(j)]["status"])
    results = np.array(results, dtype=np.int64)
    print("Total samples = {0}".format(results.shape[0]))
    def res_at_pass_k(kk=1):
        print("Results for pass@{0}".format(kk))
        vals, counts = np.unique(np.min(results[:,:kk], axis=1), return_counts=True)
        print("Success = {0} | Failure = {1} | Unparseable = {2}".format(counts[0], counts[1], counts[2]))
        print("Total unparseable = {0}".format(np.sum(results[:,:kk] == 2)))
        print("Pass rate @ {0} = {1:.3f}".format(kk, 100*counts[0]/results.shape[0]))
        print("Unparseable rate @ {0} = {1:.3f}".format(kk, 100*np.sum(results[:,:kk] == 2)/(results.shape[0]*kk)))
    res_at_pass_k(kk=1)
    res_at_pass_k(kk=2)
    res_at_pass_k(kk=5)
    return fs, results

if __name__ == "__main__":
    api_key = read_config()
    data = read_mbpp(sanitized=False)
    # mbpp_play_demo("train", 0)

    task = "Write a python function to solve the above question. No additional comments and docstrings are needed."
    prompt_file = "prompts/mbpp_prompts.txt"
    prompts = pd.read_csv(prompt_file, sep="\t", header=0, index_col="id")  # Columns are id, prompt
    prompt_id = 1
    prompt_tag = "Additional info:"
    prompt = prompts.iloc[prompt_id]["prompt"]

    if prompt_id == -1:
        final_task = task
        result_dir = "results/mbpp/no_prompt/"
    else:
        final_task = task + "\n" + prompt_tag + "\n" + prompt
        result_dir = "results/mbpp/prompt_{0}/".format(prompt_id)

    # results, status = batch_eval_0shot(data, "train", api_key, task, max_n=10, k=5, verbose=0)
    # results, status = batch_eval_fewshot(data["train"], data["prompt"], api_key, task, nshot=10, max_n=10, k=5, verbose=0)

    complete_eval_setup(data, "train", 0, result_dir, api_key, task=final_task, nshot=5, k=5, verbose=0)

    # eval_results("results/mbpp/", nshot=10, k=5)

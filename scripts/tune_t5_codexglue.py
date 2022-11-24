from library.t5_tuning import *
from rich import print

def preprocess(examples):
    # encode the code-docstring pairs
    codes = examples['code']
    docstrings = examples['docstring']
    # [SK] note: additional prompting methods can be introduced here
    prefix = "Generate Python: "
    max_input_length = 256
    max_target_length = 128
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

    inputs = [prefix + docstring for docstring in docstrings]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # encode the code responses
    labels = tokenizer(codes, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

def run():
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    example = dataset['train'][0]
    print("Code:", example["code"])
    print("Docstring:", example["docstring"])
    tune_model(dataset, preprocess)
    test_model(dataset)

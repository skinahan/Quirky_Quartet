from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import csv
import json

from library.t5_model import CodeT5
from library.t5_tuning import *
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup

# Contents adapted from:
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb

def preprocess_codexglue(examples):
    # encode the code-docstring pairs
    codes = examples['code']
    docstrings = examples['docstring']
    # [SK] note: additional prompting methods can be introduced here
    prefix = "Generate Python: "
    max_input_length = 256
    max_target_length = 128
    # tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
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

def evaluation_format(prefix_dir=''):
    answer_file = open(f"{prefix_dir}answers.json", "w")
    pred_file = open("{prefix_dir}predictions.txt", "w", encoding="utf-8")

    with open('{prefix_dir}t5_out_codex_untuned.csv', 'r', encoding="utf-8") as f_obj:
        reader_obj = csv.reader(f_obj)
        for row in reader_obj:
            if len(row) > 0:
                # Create a dict for the answer
                answer_obj = {}
                answer_obj["code"] = row[1]
                answer_obj["nl"] = row[0]

                # Output the json to the answer file
                json.dump(answer_obj, answer_file)

                # Output the prediction to the predictions file
                pred_no_newlines = row[2].replace("\n", "")
                pred_file.write(pred_no_newlines + "\n")


        f_obj.close()
    answer_file.close()
    pred_file.close()

def run():
    tune_model(dataset, process_codexglue, prefix_dir='/scratch/lsaldyt/experiments/',
               freeze=True, use_gpu=True)
    test_model(dataset, prefix_dir='/scratch/lsaldyt/experiments/')
    evaluation_format()

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import csv
import json

from library.t5_model import CodeT5
from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup


# Contents adapted from:
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb

def preprocess_examples(examples):
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


def tune_model():
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    print(dataset)

    example = dataset['train'][0]

    print("Code:", example["code"])
    print("Docstring:", example["docstring"])
    dataset = dataset.map(preprocess_examples, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8, num_workers=4)
    valid_dataloader = DataLoader(dataset['validation'], batch_size=4, num_workers=4)
    test_dataloader = DataLoader(dataset['test'], batch_size=4, num_workers=4)
    batch = next(iter(train_dataloader))
    print(batch.keys())
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    tokenizer.decode(batch['input_ids'][0])
    labels = batch['labels'][0]
    tokenizer.decode([label for label in labels if label != -100])
    model = CodeT5(train_dataloader, valid_dataloader, test_dataloader)
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    use_gpu = True
    if use_gpu:
        trainer = Trainer(gpus=1,
                          default_root_dir="./Checkpoints",
                          callbacks=[early_stop_callback, lr_monitor])
    else:
        trainer = Trainer(['cpu'],
                          default_root_dir="./Checkpoints",
                          callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model)
    # save in the current working directory, you can change this of course
    save_directory = "./"
    model.model.save_pretrained(save_directory)





def test_model():
    save_file = "./"
    #model = T5ForConditionalGeneration.from_pretrained(save_file)
    model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    test_example = dataset['test'][2]
    # print("Code:", test_example['code'])
    # prepare for the model
    input_ids = tokenizer(test_example['docstring'], return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids)
    #print("Generated code:", tokenizer.decode(outputs[0], skip_special_tokens=False))
    #print("Ground truth:", test_example['code'])
    test_set = dataset['test']

    generated_outputs = []
    ground_truth = []
    docstrings = []
    headers = ['docstring', 'label', 'output']
    with open('t5_out_codex_untuned.csv', 'a', encoding="utf-8") as f_obj:
        writer_obj = csv.writer(f_obj)
        writer_obj.writerow(headers)
        for val in test_set:
            row_list = []
            docstring = val['docstring']
            label = val['code']
            input_ids = tokenizer(val['docstring'], return_tensors='pt').input_ids
            # Using default generation setting: greedy decoding
            outputs = model.generate(input_ids, max_length=len(val['code']))
            outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
            decoded = ''.join(outputs)
            row_list.append(docstring)
            row_list.append(label)
            row_list.append(decoded)
            writer_obj.writerow(row_list)
        f_obj.close()


def evaluation_format():
    answer_file = open("answers.json", "w")
    pred_file = open("predictions.txt", "w")

    with open('t5_out_codex_tuned.csv', 'r', encoding="utf-8") as f_obj:
        reader_obj = csv.reader(f_obj)
        for row in reader_obj:
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
    # tune_model()
    test_model()

test_model()



from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import pandas as pd
import numpy as np

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
    #tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
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
    model = T5ForConditionalGeneration.from_pretrained(save_file)
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    dataset = load_dataset("code_x_glue_ct_code_to_text", "python")
    test_example = dataset['test'][2]
    #print("Code:", test_example['code'])
    # prepare for the model
    input_ids = tokenizer(test_example['docstring'], return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids)
    print("Generated code:", tokenizer.decode(outputs[0], skip_special_tokens=False))
    print("Ground truth:", test_example['code'])
    test_set = dataset['test']

    generated_outputs = []
    ground_truth = []
    docstrings = []
    for val in test_set:
        docstrings.append(val['docstring'])
        input_ids = tokenizer(val['docstring'], return_tensors='pt').input_ids
        # Using default generation setting: greedy decoding
        outputs = model.generate(input_ids, max_length=len(val['code']))
        outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in outputs]
        decoded = ''.join(outputs)
        generated_outputs.append(decoded)
        ground_truth.append(val['code'])

    ins = np.array(docstrings)
    truth = np.array(ground_truth)
    outs = np.array(generated_outputs)

    df = pd.DataFrame({"Docstrings": ins, "Labels": truth, "Output": outs})
    df.to_csv("codex_glue_text_code_eval", index=False)



def run():
    #tune_model()
    test_model()

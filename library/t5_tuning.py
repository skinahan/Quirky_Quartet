from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from pathlib import Path

from .t5_model import CodeT5
from .prompts  import to_prompt
from .checkpointer import *

def process(descriptions, source_code_list, prompt_type='Default'):
    prompts = [to_prompt(description, prompt_type=prompt_type)
               for description in descriptions]
    return postprocess(source_code_list, prompts)

def postprocess(prompts, source_code_list, tokenizer_model='Salesforce/codet5-base'):
    max_input_length = 256
    max_target_length = 128
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_model)

    model_inputs = tokenizer(prompts, max_length=max_input_length, padding="max_length", truncation=True)

    # encode the code responses
    labels = tokenizer(source_code_list, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)

    model_inputs['labels'] = labels_with_ignore_index
    return model_inputs

def tune_model(dataset, preprocess, name='t5_tuning', prefix_dir='', freeze=True, use_gpu = True):
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    if 'train' in dataset:
        train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=2,
                                      num_workers=4)
    else:
        train_dataloader = DataLoader(dataset, shuffle=True, batch_size=2,
                                      num_workers=4)
    if 'validation' in dataset:
        valid_dataloader = DataLoader(dataset['validation'], batch_size=2, num_workers=4)
    else:
        valid_dataloader = None
        print(f'No validation data provided!!')
    if 'test' in dataset:
        test_dataloader  = DataLoader(dataset['test'], batch_size=2, num_workers=4)
    else:
        test_dataloader = None
        print(f'No test data provided!!')
    batch = next(iter(train_dataloader))
    print(batch.keys())
    model = CodeT5(train_dataloader, valid_dataloader, test_dataloader, freeze=freeze)
    early_stop_callback = EarlyStopping(monitor='testing_loss', mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = CSVLogger(f'{prefix_dir}csv_data', name=name, flush_logs_every_n_steps=1)
    print(logger)

    trainer_kwargs = dict(default_root_dir=f'{name}_checkpoints',
                          callbacks=[early_stop_callback, lr_monitor],
                          logger=logger,
                          log_every_n_steps=1,
                          check_test_every_n_epoch=1,
                          max_epochs=3)
                          log_every_n_steps=1)
    if use_gpu:
        trainer = Trainer(gpus=1, **trainer_kwargs)
    else:
        trainer = Trainer(accelerator='cpu', **trainer_kwargs)
        1/0 # NO CPU :)
    trainer.fit(model)

    save_directory = Path(f'{prefix_dir}{name}/pretrained/')
    save_directory.mkdir(exist_ok=True, parents=True)
    model.model.save_pretrained(save_directory)

    if test_dataloader is not None:
        trainer.test(model, dataloaders=[test_dataloader])
    else:
        print(f'No test dataloader provided, skipping')

def test_model(dataset, name='t5_tuning', prefix_dir=''):
    save_directory = Path(f'{prefix_dir}{name}/pretrained/')
    model = T5ForConditionalGeneration.from_pretrained(save_directory)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    # tokenizer.decode(batch['input_ids'][0])
    # tokenizer.decode([label for label in labels if label != -100])
    test_example = dataset['test'][2]
    # prepare for the model
    input_ids = tokenizer(test_example['docstring'], return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids)
    print("Generated code:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Ground truth:", test_example['code'])

def test_model(dataset, name='t5_tuning', prefix_dir=''):
    save_file = f"{prefix_dir}./"
    model = T5ForConditionalGeneration.from_pretrained(save_file)
    #model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-small")
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

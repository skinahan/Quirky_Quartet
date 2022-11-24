from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from pathlib import Path

from .t5_model import CodeT5

def tune_model(dataset, preprocess, name='t5_tuning'):
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=['input_ids', 'attention_mask', 'labels'])
    train_dataloader = DataLoader(dataset['train'], shuffle=True, batch_size=8)
    valid_dataloader = DataLoader(dataset['validation'], batch_size=4)
    test_dataloader = DataLoader(dataset['test'], batch_size=4)
    batch = next(iter(train_dataloader))
    print(batch.keys())
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
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
    logger = CSVLogger('logs', name=name)
    print(logger)

    trainer_kwargs = dict(default_root_dir=f'{name}_checkpoints',
                          callbacks=[early_stop_callback, lr_monitor],
                          logger=[logger])
    # use_gpu = True
    use_gpu = False
    if use_gpu:
        trainer = Trainer(gpus=1, **trainer_kwargs)
    else:
        trainer = Trainer(accelerator='cpu', **trainer_kwargs)
    trainer.fit(model)

    save_directory = Path(f'{name}/pretrained')
    save_directory.mkdir(exist_ok=True)

    model.model.save_pretrained(save_directory)

def test_model(dataset):
    model = T5ForConditionalGeneration.from_pretrained(save_directory)
    test_example = dataset['test'][2]
    print("Code:", test_example['code'])
    # prepare for the model
    input_ids = tokenizer(test_example['docstring'], return_tensors='pt').input_ids
    # generate
    outputs = model.generate(input_ids)
    print("Generated code:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Ground truth:", test_example['code'])

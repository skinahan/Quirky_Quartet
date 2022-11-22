from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl

# Contents adapted from:
# https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb

class CodeT5(pl.LightningModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader, lr=5e-5, num_train_epochs=15,
                 warmup_steps=1000):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
        self.save_hyperparameters()
        self.training_dataloader = train_dataloader
        self.valid_dataloader = val_dataloader
        self.testing_dataloader = test_dataloader

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(self.training_dataloader)
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.hparams.warmup_steps,
                                                                     num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        return self.training_dataloader

    def val_dataloader(self):
        return self.valid_dataloader

    def test_dataloader(self):
        return self.testing_dataloader

import torch
import transformers
import pytorch_lightning as pl
import torchmetrics


class Model(pl.LightningModule):
    def __init__(
        self,
        model_name,
        lr,
        max_epoch,
        warmup_ratio,
        gradient_accumulation_steps,
        tokenizer,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = model_name
        self.lr = lr
        self.max_epoch = max_epoch
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name, num_labels=1
        )
        self.plm.resize_token_embeddings(len(tokenizer))

        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        x = self.plm(x)["logits"]

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)
        self.log(
            "val_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log(
            "test_pearson",
            torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()),
        )
        return logits, y

    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)

        return logits.squeeze()

    def configure_optimizers(self):
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.gradient_accumulation_steps
            * self.max_epoch
        )
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps,
        )

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

        return optimizer_config

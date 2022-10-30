import argparse
import os
import re
import emoji
import random
import numpy as np
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import pandas as pd

from tqdm.auto import tqdm

import transformers
import torch
import torchmetrics
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from soynlp.normalizer import repeat_normalize

import dill

import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
pattern = re.compile(f"[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+")
url_pattern = re.compile(
    r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    # 학습 및 추론 과정에서 데이터를 1개씩 꺼내오는 곳
    def __getitem__(self, idx):
        # 정답이 있다면 else문을, 없다면 if문을 수행합니다
        if len(self.targets) == 0:
            return torch.tensor(self.inputs[idx])
        else:
            return torch.tensor(self.inputs[idx]), torch.tensor(self.targets[idx])

    # 입력하는 개수만큼 데이터를 사용합니다
    def __len__(self):
        return len(self.inputs)


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        dev_path,
        test_path,
        predict_path,
        num_workers,
        tokenizer,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = tokenizer

        self.target_columns = ["label"]
        self.delete_columns = ["id"]
        self.text_columns = ["sentence_1", "sentence_2"]

    def text_cleaning(self, text):
        # Ref: https://github.com/Beomi/KcBERT
        text = pattern.sub(" ", text)
        text = emoji.replace_emoji(text, replace="")
        text = url_pattern.sub("", text)
        text = text.strip()
        text = repeat_normalize(text, num_repeats=2)
        return text

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = "[SEP]".join(
                [item[text_column] for text_column in self.text_columns]
            )
            outputs = self.tokenizer(
                text, add_special_tokens=True, padding="max_length", truncation=True
            )
            data.append(outputs["input_ids"])
        return data

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # Text Cleaning
        for text_column in self.text_columns:
            data[text_column] = data[text_column].apply(self.text_cleaning)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage="fit"):
        if stage == "fit":
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=args.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


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


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    torch.cuda.empty_cache()
    torch.manual_seed(404)


    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="beomi/KcELECTRA-base", type=str)
    parser.add_argument("--wandb_label", default="UNK test", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=3, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--wandb_offline", default=False, type=bool)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--train_path", default="../data/train.csv", type=str)
    parser.add_argument("--dev_path", default="../data/dev.csv", type=str)
    parser.add_argument("--test_path", default="../data/dev.csv", type=str)
    parser.add_argument("--predict_path", default="../data/test.csv", type=str)
    parser.add_argument("--random_seed", default=404, type=int)
    args = parser.parse_args()

    # Fix Seed
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed) # if multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # wandb setup
    wandb_name = f"{args.model_name}_lr_{args.learning_rate}_{args.wandb_label}"

    wandb_logger = WandbLogger(
        name=wandb_name, project="mySTS", offline=args.wandb_offline
    ) # entity='ecl-mlstudy
    wandb_logger.experiment.config.update(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, max_length=160
    )
    extra_tokens = ['빂', '칻', '뿨' ,'ᆢ', '탆', '믱', '👌' , '☼' ,'｀', '뵛' ,'굠', '큩', '훃' ,'즠']
    tokenizer.add_tokens(extra_tokens)

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        args.shuffle,
        args.train_path,
        args.dev_path,
        args.test_path,
        args.predict_path,
        args.num_workers,
        tokenizer,
    )

    model = Model(
        args.model_name,
        args.learning_rate,
        args.max_epoch,
        args.warmup_ratio,
        args.gradient_accumulation_steps,
        tokenizer,
    )

    # callbacks custimization
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2, mode="max", monitor="val_pearson"
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    
    
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        callbacks=[lr_monitor, checkpoint_callback, early_stop_callback],
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, "model.pt", pickle_module=dill)
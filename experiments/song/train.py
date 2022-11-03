import argparse

import transformers
import torch
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
)


import dill
import pandas as pd

from dataloader import Dataloader
from model import Model

if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="monologg/koelectra-base-v3-discriminator", type=str
    )
    parser.add_argument("--wandb_label", default="", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=30, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--wandb_offline", default=False, type=bool)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--train_path", default="../../data/train.csv", type=str)
    parser.add_argument("--dev_path", default="../../data/dev.csv", type=str)
    parser.add_argument("--test_path", default="../../data/dev.csv", type=str)
    parser.add_argument("--predict_path", default="../../data/test.csv", type=str)
    parser.add_argument("--new_token_path", default="new_token.csv", type=str)
    args = parser.parse_args()

    wandb_name = (
        f"Robin 10 epochs no punctuations"
    )

    wandb_logger = WandbLogger(
        name=wandb_name, entity="ecl-mlstudy", project="STS", offline=args.wandb_offline
    )
    wandb_logger.experiment.config.update(args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, max_length=160
    )

    new_tokens = pd.read_csv(args.new_token_path).columns.tolist()

    tokenizer.add_tokens(new_tokens)
    special_tokens_dict = {"additional_special_tokens": ["[RTT]", "[ORG]"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    pl.seed_everything(args.seed)

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


    early_stop_callback = EarlyStopping(
        monitor="val_pearson",
        min_delta=0.00,
        patience=5,
        mode="max",
    )
    

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=args.max_epoch,
        log_every_n_steps=1,
        callbacks=[
            lr_monitor,
            early_stop_callback,       
            ],
        accumulate_grad_batches=args.gradient_accumulation_steps,
    )

    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    torch.save(model, f"{args.seed}_KoELECTRA_base.pt", pickle_module=dill)

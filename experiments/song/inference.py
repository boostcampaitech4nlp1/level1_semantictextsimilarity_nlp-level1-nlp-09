import argparse
import pandas as pd

import transformers
import torch
import pytorch_lightning as pl

from soynlp.normalizer import repeat_normalize
import dill
from model import Model
from dataloader import Dataloader


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="monologg/koelectra-base-v3-discriminator", type=str
    )
    parser.add_argument("--wandb_label", default="", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epoch", default=35, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=404, type=int)
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
    args = parser.parse_args(args=[])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name, max_length=160
    )
    new_tokens = pd.read_csv(args.new_token_path).columns.tolist()
    tokenizer.add_tokens(new_tokens)
    special_tokens_dict = {"additional_special_tokens": ["[RTT]", "[ORG]"]}
    tokenizer.add_special_tokens(special_tokens_dict)

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

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epoch, log_every_n_steps=1)

    # Inference part
    # 저장된 모델로 예측을 진행합니다.
    checkpoint = torch.load(
        "STS/nfx3ehoo/checkpoints/epoch=19-step=5840.ckpt",
        pickle_module=dill,
    )
    # print(checkpoint.keys())

    # model = torch.load("model_10_28.pt", pickle_module=dill)
    model.load_state_dict(
        checkpoint["state_dict"],
    )
    trainer.test(model=model, datamodule=dataloader)
    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
    output = pd.read_csv("../../data/test.csv")
    output["target"] = predictions
    output.to_csv("output.csv", index=False)


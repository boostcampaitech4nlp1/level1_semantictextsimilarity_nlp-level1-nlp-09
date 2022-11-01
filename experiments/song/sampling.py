import argparse
from kortt import Generator
import pandas as pd
import math
from tqdm import tqdm


def get_grouped_label_df(df):
    df["group_label"] = df["label"].apply(math.floor)
    df = df.groupby("group_label")
    return df


def print_df_group_cnt(grouped_df):
    print("----------Count by group_label----------------")
    print(grouped_df["id"].count())
    print("----------------------------------------------")


def get_sampling(target_df, ratio, generator, augmentation_label):
    sample_num = int(len(target_df) * ratio)
    sample_df = target_df.sample(sample_num)
    for i, row in tqdm(
        sample_df.iterrows(),
        total=len(sample_df),
        desc=f"Label {augmentation_label}",
    ):
        row["sentence_1"] = generator.generate(row["sentence_1"])
        sample_df.loc[i] = row
    return sample_df


def get_augmentation_df(grouped_df, augmentation_labels, augemtation_ratios, generator):
    if len(augmentation_labels) != len(augemtation_ratios):
        raise Exception(" ratio list length MUST BE SAME label list length")

    for augmentation_label, augemtation_ratio in zip(
        augmentation_labels, augemtation_ratios
    ):
        if not (augmentation_label >= 0 and augmentation_label < 5):
            raise Exception("augmentation_label must be 0-5")
        if not (type(augmentation_label) == int):
            raise Exception("augmentation_label type must be int")

        target_df = grouped_df.get_group(augmentation_label)
        sample_df = get_sampling(
            target_df, augemtation_ratio, generator, augmentation_label
        )
        print(sample_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="../../data/train.csv", type=str)
    parser.add_argument("--dev_path", default="../../data/dev.csv", type=str)
    args = parser.parse_args()

    generator = Generator(mode="google")
    augmentation_labels = [1, 3, 5]
    augemtation_ratios = [
        0.5,
        0.8,
        0.3,
    ]  # ratio list length MUST BE SAME label list length!
    print(len(augmentation_labels), len(augemtation_ratios))
    train_df = pd.read_csv(args.train_path)
    grouped_train_df = get_grouped_label_df(train_df)
    print_df_group_cnt(grouped_train_df)
    train_augmentation_df = get_augmentation_df(
        grouped_train_df, augmentation_labels, augemtation_ratios, generator
    )

import os
import json
import pickle
import argparse
import pandas as pd
import numpy as np
import datasets
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import evaluate
from transformers import AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple choice questions on a certain model")

    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    args = parser.parse_args()

    """
    # Sanity checks
    #if args.task_name is None and args.train_file is None and args.validation_file is None:
    #    raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
    """

    return args


args = parse_args()

# Load in datasets
if args.train_file is not None:
    train_df = pd.read_csv(args.train_file).dropna().drop_duplicates()
    test_df = pd.read_csv(args.validation_file).dropna().drop_duplicates()

    # only grab MC questions for now
    train_df = train_df.loc[train_df['qtype'] == "mc"]
    test_df = test_df.loc[test_df['qtype'] == "mc"]

    # Load into dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    # cc_news = load_dataset('cc_news', split="train")
    raw_dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    label_list = raw_dataset["train"].unique("label")
    num_labels = len(label_list)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    # Pre-process labels
    label_list.sort()
    label_to_id = {key: value for value, key in enumerate(label_list)}

    def process_data(examples):
        """
        This function processes the input data using a tokenizer to prepare for the model.
        Also, updates the label values to match model config
        :param examples: DataSet
        :return:
        """

        result = tokenizer(examples["question"], padding=True, truncation=True)
        if "label" in examples:
            if label_to_id is not None:
                # Map labels to IDs (not necessary for GLUE tasks)
                result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    processed_datasets = raw_dataset.map(process_data, batched=True, desc="Running Tokenizer on data")

    training_args = TrainingArguments(output_dir=".", evaluation_strategy="no")
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Train model
    trainer = Trainer(model=model, args=training_args, train_dataset=processed_datasets["train"], compute_metrics=compute_metrics)
    # trainer.train()

    preds = trainer.predict(processed_datasets["test"])

    print(preds)










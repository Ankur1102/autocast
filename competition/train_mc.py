import ast
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
from Collator import DataCollatorForMultipleChoice
from transformers import AutoModel, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForMultipleChoice, BigBirdForMultipleChoice
from helpers import *


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

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
       raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."
    return args


args = parse_args()

# Load in datasets
autocast_questions = json.load(open('train_dataset.json', encoding="utf-8"))  # from the Autocast dataset
test_questions = json.load(open('test_dataset.json', encoding="utf-8"))
train_data = pd.DataFrame(autocast_questions).dropna()
test_data = pd.DataFrame(test_questions).dropna()
# train_data = pd.read_json(args.train_file).dropna()
# test_data = pd.read_json(args.validation_file).dropna()
train_data = train_data.loc[train_data.astype(str).drop_duplicates().index]
test_data = test_data.loc[test_data.astype(str).drop_duplicates().index]

# only grab MC questions for now
mc_df = train_data.loc[train_data['qtype'] == "mc"]
mc_df = mc_df.rename(columns={'answer': 'label'})
# Split training data, to get an evaluation set
label_list = mc_df.label.unique()
train_df = mc_df.sample(frac=0.8, random_state=25)
eval_df = mc_df.drop(train_df.index)
test_df = test_data.loc[test_data['qtype'] == "mc"]

# Load into dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)
cc_news_dataset = load_dataset('cc_news', split="train")
raw_dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset, "eval": eval_dataset})

num_labels = len(label_list)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModel.from_pretrained(args.model_name_or_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token

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

    num_examples = len(examples["question"])
    num_choices = num_labels # Force num_choices to match num_labels
    result = {}
    for i in range(num_examples):
        choices = examples["choices"][i]
        pad = ["other"] * (num_labels - len(choices))
        choices.extend(pad)
        test = list(zip([examples["question"][i]] * num_choices, choices))
        sentence = tokenizer.batch_encode_plus(test, truncation=True)
        for k, v in sentence.items():
            if k not in result:
                result[k] = []
            result[k].append(v)
    if "label" in examples:
        if label_to_id is not None:
            # Map labels to IDs (not necessary for GLUE tasks)
            result["label"] = [label_to_id[l] for l in examples["label"]]
    return result


def process_news_data(examples):

    if "text" in examples:
        result = tokenizer(examples["text"], padding='max_length', truncation=True)
    return result


# Train model with cc news
processed_news = cc_news_dataset.map(process_news_data, batched=True, desc="Running Tokenizer on data")
reduced = processed_news.remove_columns([x for x in processed_news.column_names
                                         if (x != "input_ids" and x != "attention_mask")])
data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
news_dataloader = DataLoader(reduced.select(range(10)), shuffle=True, batch_size=8, collate_fn=data_collator)
for epoch in range(1):
    model.train()
    for step, batch in enumerate(tqdm(news_dataloader)):
        outputs = model(**batch)
tuned_trainer = Trainer(model=model)
model_path = os.path.join(".", args.output_dir, "cc_bert")
tuned_trainer.save_model(output_dir=model_path)

processed_datasets = raw_dataset.map(process_data, batched=True, desc="Running Tokenizer on data")

# Evaluate cc_news_bert
cc_bert = AutoModelForMultipleChoice.from_pretrained(model_path, num_labels=num_labels)
tuned_args = TrainingArguments(output_dir="cc_bert_output", num_train_epochs=10, learning_rate=2e-5,
                               logging_steps=16, save_steps=16, eval_steps=16, save_total_limit=2,
                               metric_for_best_model='accuracy',
                               greater_is_better=True, load_best_model_at_end=True,
                               evaluation_strategy="steps", per_device_train_batch_size=2,
                               gradient_accumulation_steps=8, gradient_checkpointing=True)
metric = evaluate.load("accuracy")
cc_bert_trainer = Trainer(model=cc_bert, args=tuned_args,
                          train_dataset=processed_datasets["train"],
                          eval_dataset=processed_datasets["eval"],
                          data_collator=DataCollatorForMultipleChoice(tokenizer), compute_metrics=compute_metrics)

metrics = cc_bert_trainer.evaluate()
cc_bert_trainer.log_metrics("eval", metrics)

# Fine tune trained model with train set
tuned_model = AutoModelForMultipleChoice.from_pretrained(model_path, num_labels=num_labels)
tuned_args = TrainingArguments(output_dir=".", logging_steps=24, save_steps=24, eval_steps=24, save_total_limit=2, metric_for_best_model='accuracy',
                           greater_is_better=True, load_best_model_at_end=True,
                           evaluation_strategy="steps", per_device_train_batch_size=2,
                           gradient_accumulation_steps=8, gradient_checkpointing=True)
metric = evaluate.load("accuracy")
fine_tune_trainer = Trainer(model=tuned_model, args=tuned_args, train_dataset=processed_datasets["train"].select(range(10)),
                            eval_dataset=processed_datasets["eval"].select(range(10)),
                            data_collator=DataCollatorForMultipleChoice(tokenizer), compute_metrics=compute_metrics)
fine_tune_trainer.train()
model_path = os.path.join(".", args.output_dir, "fine_tune_bert")
fine_tune_trainer.save_model()
# preds = trainer.predict(processed_datasets["test"])
#
# print(preds)










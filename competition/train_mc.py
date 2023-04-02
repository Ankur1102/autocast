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
from Collator import *
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
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--fewshot",
        type=bool,
        default=False,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--predict",
        type=bool,
        default=False,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--train_batch",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
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
    return args


args = parse_args()

# Load in datasets
autocast_questions = json.load(open('train_dataset.json', encoding="utf-8"))  # from the Autocast dataset
test_questions = json.load(open('test_dataset.json', encoding="utf-8"))
# Drop indices with nan
train_data = pd.DataFrame(autocast_questions).dropna()
test_data = pd.DataFrame(test_questions).dropna()
# Drop duplicates
train_data = train_data.loc[train_data.astype(str).drop_duplicates().index]
test_data = test_data.loc[test_data.astype(str).drop_duplicates().index]

# only grab MC questions for now
mc_df = train_data.loc[train_data['qtype'] == "mc"]
mc_df = mc_df.rename(columns={'answer': 'label'})
label_list = mc_df.label.unique()
num_labels = len(label_list)
# Delete data points that violate max model length
ind = []
for i, (x, y) in enumerate(zip(mc_df["question"], mc_df["choices"])):
    # Get an average length for the choices
    avg_choice_length = sum(sum([[len(item.split(" "))] for item in y], []))/len(y)
    # Need question length + choice length to not exceed 512 for model sequence
    if (len(x.split(" ")) + avg_choice_length + 15) > 512/len(y):
        ind.append(i)
mc_df.drop(mc_df.iloc[ind].index, inplace=True)
# Split training data, to get an evaluation set
train_df = mc_df.sample(frac=0.8, random_state=25)
eval_df = mc_df.drop(train_df.index)
test_df = test_data.loc[test_data['qtype'] == "mc"]
ind = []
for i, (x, y) in enumerate(zip(test_df["question"], test_df["choices"])):
    # Get an average length for the choices
    avg_choice_length = sum(sum([[len(item.split(" "))] for item in y], []))/len(y)
    # Need question length + choice length to not exceed 512 for model sequence
    if (len(x.split(" ")) + avg_choice_length + 15) > 512/len(y):
        ind.append(i)
test_df.drop(test_df.iloc[ind].index, inplace=True)

# Load into dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)
cc_news_dataset = load_dataset('cc_news', split="train")
raw_dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset, "eval": eval_dataset})

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
    num_choices = num_labels  # Force num_choices to match num_labels
    
    result = {}
    for i in range(num_examples):
        choices = examples["choices"][i]
        num_choices = len(choices)
        # pad = ["other"] * (num_labels - len(choices))
        # choices.extend(pad)
        test = list(zip([examples["question"][i]] * num_choices, choices))
        sentence = tokenizer.batch_encode_plus(test, padding='longest', truncation=True, return_token_type_ids=False,
                                               return_attention_mask=False)
        input_ids = sum([sum(v, []) for k, v in sentence.items()], [])
        flat_sentence = tokenizer.prepare_for_model(input_ids, padding='longest', truncation=True)
        for k, v in flat_sentence.items():
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


def verify_process_function(raw_dataset, num_elements, skip=True):
    """
    This function will print out the dimensions of the process_data function.
    Followed by a print of the decoded input
    """
    if not skip:
        # Check process function
        examples = raw_dataset["train"].select(range(5))
        features = process_data(examples)
        print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])
        print([tokenizer.decode(features["input_ids"][0][i]) for i in range(len(features["input_ids"]))])


def verify_collator_function(processed_dataset, num_elements, skip=True):
    """
    This function prints out the output of the DataCollator( decoded tokenized input).
    Followed by the same element but directly from the raw dataset
    """
    if not skip:
        # check collator
        accepted_keys = ["input_ids", "attention_mask", "label"]
        features = [{k: v for k, v in processed_dataset["train"][i].items() if k in accepted_keys} for i in range(num_elements)]
        batch = DataCollatorForMultipleChoice(tokenizer)(features)
        print([tokenizer.decode(batch["input_ids"][8][i].tolist()) for i in range(10)])
        show_one(raw_dataset["train"][8])


def show_one(example):
    print(f"Question: {example['question']}")
    choices = [choice for choice in example['choices']]
    for choice in choices:
        print(choice)
    # print(f"Label is {example['answer']}")


def generate_train_args(output_dir):
    """
    Generates and returns training arguments
    """

    if args.train_steps is None:
        return TrainingArguments(output_dir=output_dir, num_train_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate, per_device_train_batch_size=args.train_batch,
                                 per_device_eval_batch_size=args.eval_batch, evaluation_strategy="epoch",
                                 gradient_accumulation_steps=args.gradient_steps, gradient_checkpointing=True,
                                 metric_for_best_model='accuracy', logging_steps=(args.gradient_steps*2))
    else:
        return TrainingArguments(output_dir=output_dir, num_train_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate, logging_steps=args.train_steps,
                                 save_steps=args.train_steps, eval_steps=args.train_steps, save_total_limit=2,
                                 metric_for_best_model='accuracy', greater_is_better=True, load_best_model_at_end=True,
                                 evaluation_strategy="steps", per_device_train_batch_size=args.train_batch,
                                 gradient_accumulation_steps=args.gradient_steps, gradient_checkpointing=True)


processed_datasets = raw_dataset.map(process_data, batched=True, desc="Running Tokenizer on data")

# Train model with cc news
if args.fewshot:
    processed_news = cc_news_dataset.map(process_news_data, batched=True, desc="Running Tokenizer on data")
    reduced = processed_news.remove_columns([x for x in processed_news.column_names
                                             if (x != "input_ids" and x != "attention_mask")])
    data_collator = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    news_dataloader = DataLoader(reduced.select(range(10)), shuffle=True, batch_size=args.train_batch,
                                 collate_fn=data_collator)
    for epoch in range(1):
        model.train()
        for step, batch in enumerate(tqdm(news_dataloader)):
            outputs = model(**batch)
    tuned_trainer = Trainer(model=model)
    model_path = os.path.join(".", args.output_dir, "cc_bert")
    tuned_trainer.save_model(output_dir=model_path)


    # # Evaluate cc_news_bert
    cc_bert = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tuned_args = generate_train_args("cc_bert")
    metric = evaluate.load("accuracy")
    cc_bert_trainer = Trainer(model=cc_bert, args=tuned_args,
                              train_dataset=processed_datasets["train"],
                              eval_dataset=processed_datasets["eval"], compute_metrics=compute_metrics)

    metrics = cc_bert_trainer.evaluate()
    cc_bert_trainer.log_metrics("eval", metrics)
#
# # # Fine cc_news_bert on t\f questions
# tuned_model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
# tuned_args = TrainingArguments(output_dir=".", logging_steps=24, save_steps=24, eval_steps=24, save_total_limit=2, metric_for_best_model='accuracy',
#                            greater_is_better=True, load_best_model_at_end=True,
#                            evaluation_strategy="steps", per_device_train_batch_size=2,
#                            gradient_accumulation_steps=8, gradient_checkpointing=True)
# metric = evaluate.load("accuracy")
# fine_tune_trainer = Trainer(model=tuned_model, args=tuned_args, train_dataset=processed_datasets["train"].select(range(10)),
#                             eval_dataset=processed_datasets["eval"].select(range(10)), compute_metrics=compute_metrics)
# fine_tune_trainer.train()
# model_path = os.path.join(".", args.output_dir, "fine_tune_bert_tf")
# fine_tune_trainer.save_model()
#

# Fine tune trained model with train set
tuned_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
model_path = os.path.join(".", args.output_dir, "fine_tune_bert_mc")
tuned_args = generate_train_args(model_path)
metric = evaluate.load("accuracy")
fine_tune_trainer = Trainer(model=tuned_model, args=tuned_args, train_dataset=processed_datasets["train"],
                            eval_dataset=processed_datasets["eval"],
                            data_collator=DataCollatorForSequenceClassification(tokenizer), compute_metrics=compute_metrics)
metrics = fine_tune_trainer.train()
fine_tune_trainer.log_metrics("all", metrics)
fine_tune_trainer.save_metrics("all", metrics)

fine_tune_trainer.save_model()
if args.predict:
    preds = fine_tune_trainer.predict()











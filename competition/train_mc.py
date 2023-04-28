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
from scipy.special import softmax
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import evaluate
from Collator import *
import GPUtil
from transformers import AutoModel, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, \
    AutoModelForMultipleChoice, BigBirdForMultipleChoice, pipeline


def parse():
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
        "--train",
        action="store",
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


def cleanData(train_df, test_df):
    """
    Takes in a train, test df file and runs steps to clean data
    """
    # Drop duplicates
    train_df = train_df.loc[train_df.astype(str).drop_duplicates().index]
    test_df = test_df.loc[test_df.astype(str).drop_duplicates().index]
    
    # Delete rows where the question is only numbers
    test_df = test_df[test_df["question"].apply(lambda x: pd.to_numeric(x, errors='coerce')).isna()]
    train_df = train_df[train_df["question"].apply(lambda x: pd.to_numeric(x, errors='coerce')).isna()]

    # Drop unwanted columns
    train_df.drop(["prediction_count", "forecaster_count", "crowd"], axis=1, inplace=True)

    # Get test df ids and make sure there is no overlap between train & test data
    test_id = test_df['id'].tolist()
    train_id = train_df['id'].tolist()
    common_ind = [id for id in train_id if id in test_id]
    train_df = train_df[~train_df.id.isin(common_ind)]

    return train_df, test_df


args = parse()
# Load in datasets
train_questions = json.load(open('train_dataset.json', encoding="utf-8"))  # from the Autocast dataset
test_questions = pd.read_csv('autocast_test_set_w_answers.csv', index_col=0, converters={"choices": eval})

#test_questions = json.load(open('test_dataset.json', encoding="utf-8"))

# Read in train set using pandas and automatically drop nan rows
train_df = pd.DataFrame(train_questions).dropna()
test_df = test_questions.dropna()

# CLean data from json files
train_data, test_data = cleanData(train_df, test_df)

# only grab MC questions for now
mc_df = train_data.loc[train_data['qtype'] != "num"]
mc_df = mc_df.rename(columns={'answer': 'label'})
label_list = mc_df.label.unique()
num_labels = len(label_list)
# Split training data, to get an evaluation set
train_df = mc_df.sample(frac=0.8, random_state=25)
eval_df = mc_df.drop(train_df.index)
test_df = test_data.loc[test_data['qtype'] != "num"]
# test_df = test_df.rename(columns={'answers': 'label'})
# Load into dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
test_dataset = Dataset.from_pandas(test_df)
cc_news_dataset = load_dataset('cc_news', split="train")
raw_dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset, "eval": eval_dataset})

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=num_labels)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id


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
        question = examples["question"][i]
        background = examples["background"][i]
        back_ion = [background + question]
        # pairs = [[f"{background} {question} {choice}"] for choice in choices]
        # pairs = sum(pairs, [])
        pairs = tuple(zip(back_ion * num_choices, choices))
        sentence = tokenizer.batch_encode_plus(pairs, add_special_tokens=True, padding="longest", truncation=False, return_token_type_ids=False,
                                               return_attention_mask=False, pad_to_multiple_of=9)
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
    answer = example["label"]
    print(f"Actual answer is {answer}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def generate_train_args(output_dir):
    """
    Generates and returns training arguments
    """

    if args.train_steps is None:
        return TrainingArguments(output_dir=output_dir, optim='adamw_torch', weight_decay=0.1, num_train_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate, per_device_train_batch_size=args.train_batch,
                                 per_device_eval_batch_size=args.eval_batch, evaluation_strategy="epoch",
                                 gradient_accumulation_steps=args.gradient_steps, gradient_checkpointing=False,
                                 metric_for_best_model='accuracy', logging_steps=args.gradient_steps)
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


    # Evaluate cc_news_bert
    cc_bert = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
    tuned_args = generate_train_args("cc_bert")
    metric = evaluate.load("accuracy")
    cc_bert_trainer = Trainer(model=cc_bert, args=tuned_args,
                              train_dataset=processed_datasets["train"],
                              eval_dataset=processed_datasets["eval"], compute_metrics=compute_metrics,
                              data_collator=DataCollatorForSequenceClassification(tokenizer))

    metrics = cc_bert_trainer.evaluate()
    cc_bert_trainer.log_metrics("eval", metrics)


# Fine tune trained model with train set
model_path = os.path.join(".", args.output_dir, "GPT")
model.config.hidden_dropout_prob = 0.5
model.config.attention_probs_dropout_prob = .5
tuned_args = generate_train_args(model_path)
metric = evaluate.load("accuracy")
fine_tune_trainer = Trainer(model=model, args=tuned_args, train_dataset=processed_datasets["train"],
                            eval_dataset=processed_datasets["train"],
                            data_collator=DataCollatorForSequenceClassification(tokenizer), compute_metrics=compute_metrics)
if False:
    metrics = fine_tune_trainer.train()
    fine_tune_trainer.save_model()
    fine_tune_trainer.log_metrics("train", metrics.metrics)
    train_metrics = fine_tune_trainer.evaluate(eval_dataset=processed_datasets["eval"])
    fine_tune_trainer.log_metrics("eval", train_metrics)
    # metrics = fine_tune_trainer.evaluate()
    # fine_tune_trainer.log_metrics("eval", metrics)

if args.predict:
    preds = fine_tune_trainer.predict(test_dataset=processed_datasets["test"])
    pred_df = pd.DataFrame(preds.predictions)
    save_path = os.path.join(model_path, "predictions.csv")
    pred_df.to_csv(save_path)
    preds = preds.predictions
    # preds_df = pd.read_csv(os.path.join(model_path, "predictions.csv"), index_col=0)
    # preds = preds_df.to_numpy()
    preds = softmax(preds, axis=1)

    # Run local evaluation
    def brier_score(probabilities, answer_probabilities):
        return ((probabilities - answer_probabilities) ** 2).sum() / 2

    # Read in answers
    answers = []
    qtypes = []
    for question in processed_datasets["test"]:
        if question['qtype'] == 't/f':
            # No [1,0]; Yes [0,1]
            ans_idx = 0 if question['answers'] == 'no' else 1
            ans = np.zeros(len(question['choices']))
            ans[ans_idx] = 1
            qtypes.append('t/f')
        elif question['qtype'] == 'mc':
            ans_idx = ord(question['answers']) - ord('A')
            ans = np.zeros(len(question['choices']))
            ans[ans_idx] = 1
            qtypes.append('mc')
        elif question['qtype'] == 'num':
            ans = float(question['answers'])
            qtypes.append('num')
        answers.append(ans)

    tf_results, mc_results, num_results = [], [], []
    for i, (p, a, qtype) in enumerate(zip(preds, answers, qtypes)):
        if qtype == 't/f':
            p = p[10:]
            tf_results.append(brier_score(p, a))
        elif qtype == 'mc':
            if i>=478:
                test = 1
            index = len(processed_datasets["test"][i]["choices"])
            if index > 12:
                index = 12
            p = p[:index]
            mc_results.append(brier_score(p, a))
        else:
            num_results.append(np.abs(p - a))
        print(i)

    print(
        f"T/F: {np.mean(tf_results) * 100:.2f}, MCQ: {np.mean(mc_results) * 100:.2f}, NUM: {np.mean(num_results) * 100:.2f}")
    print(f"Combined Metric: {(np.mean(tf_results) + np.mean(mc_results) + np.mean(num_results)) * 100:.2f}")













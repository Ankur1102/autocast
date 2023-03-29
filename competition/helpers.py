from Collator import DataCollatorForMultipleChoice

def show_one(example):
    print(f"Question: {example['question']}")
    choices = [choice for choice in example['choices']]
    for choice in choices:
        print(choice)
    # print(f"Label is {example['answer']}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def verify_process_function(raw_dataset, num_elements, skip=True):
    """
    This function will print out the dimensions of the process_data function. 
    Followed by a print of the decoded input 
    """
    if not skip:
        # Check process function
        examples = raw_dataset["train"][:num_elements]
        features = process_data(examples)
        print(len(features["input_ids"]), len(features["input_ids"][0]), [len(x) for x in features["input_ids"][0]])
        print([tokenizer.decode(features["input_ids"][0][i]) for i in range(4)])

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
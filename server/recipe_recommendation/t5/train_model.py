import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from datasets import load_dataset, Dataset, load_metric
import numpy as np
from datetime import datetime

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

dataset_train_validation = dataset['train'].train_test_split(test_size=0.2)
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']

# Filter out examples with missing or empty ingredients
dataset = dataset.filter(lambda example: example['ingredients'] is not None and len(example['ingredients']) > 20)

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))

# Tokenize inputs and targets separately
ingredients_prefix = "ingredients: "
title_prefix = "title: "
directions_prefix = "directions: "
def preprocess_data(examples):
    inputs = [ingredients_prefix + text for text in examples["ingredients"]]
    titles = [title_prefix + text for text in examples["title"]]
    directions = [directions_prefix + text for text in examples["directions"]]

    # Combine titles and directions
    title_directions = [title + " " + directions for title, directions in zip(titles, directions)]
    model_inputs = tokenizer(inputs, max_length=300, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(title_directions, max_length=600, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True)

print("tokenized dataset: ", tokenized_datasets)

batch_size = 4

current_datetime = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
output_dir = f"server/recipe_recommendation/t5/models/t5-small-fine-tuned_{current_datetime}"

training_args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=1e-3, # adjust
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01, 
    save_total_limit=3,
    num_train_epochs=10, # adjust
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    report_to="tensorboard",
    gradient_accumulation_steps=2
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

# Load metrics
rouge_metric = load_metric("rouge")
bleu_metric = load_metric("bleu")

def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    rouge_output = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    #bleu_output = bleu_metric.compute(predictions=decoded_preds, references=[decoded_labels])
    
    rouge_results = {key: value.mid.fmeasure for key, value in rouge_output.items()}
    #bleu_results = bleu_output["bleu"]
    
    #return {**rouge_results, **bleu_results}
    return {**rouge_results}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
    ##push.weight_and_bias
)

print("Training the model...")
trainer.train()

# Save the model
trainer.save_model = output_dir
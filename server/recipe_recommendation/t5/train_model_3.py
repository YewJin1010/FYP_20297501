import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, load_metric
import numpy as np
import nltk
nltk.download('punkt')
from transformers import AdamW, get_linear_schedule_with_warmup, EarlyStoppingCallback
from torch.optim.lr_scheduler import LambdaLR


# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

dataset_train_validation = dataset['train'].train_test_split(test_size=0.2)
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))

# Filter out examples with missing or empty ingredients
dataset = dataset.filter(lambda example: example['ingredients'] is not None and len(example['ingredients']) > 0)

# Tokenize inputs and targets separately
prefix = "ingredients: "
def preprocess_data(examples):
    inputs = [prefix + text for text in examples["ingredients"]]
    model_inputs = tokenizer(inputs, max_length=300, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["directions"], max_length=600, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.filter(lambda example: example['ingredients'] is not None and len(example['ingredients']) > 0)
tokenized_datasets = dataset.map(preprocess_data, batched=True)

print("tokenized dataset: ", tokenized_datasets)

batch_size = 4
output_dir = "server/recipe_recommendation/t5/models/t5-small-conditional-generation"

# Load the ROUGE metric
rouge_metric = load_metric("rouge")

training_args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=1e-1, # adjust
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

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

total_steps = len(tokenized_datasets['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-3)

# Define scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,  # Adjust warmup steps as needed
    num_training_steps= total_steps
)

# Define gradient clipping
if training_args.max_grad_norm is not None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    optimizers=(optimizer, scheduler)
)

print("Training the model...")
trainer.train()

# Save the model
trainer.save_model = output_dir
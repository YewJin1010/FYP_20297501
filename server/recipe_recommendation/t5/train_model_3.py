import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset, load_metric

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

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
output_dir = "server/recipe_recommendation/t5/models/t5-base-conditional-generation-nolimit"

training_args = Seq2SeqTrainingArguments(
    output_dir = output_dir,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=1e-3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01, 
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=False,
    report_to="tensorboard",
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained("t5-base")

trainer = Seq2SeqTrainer(
    model_init=model_init,
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer
)

print("Training the model...")
# Train the model
trainer.train()

# Save the model
trainer.save_model = output_dir
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the pre-trained GPT-2 model for sequence-to-sequence tasks
model = GPT2LMHeadModel.from_pretrained("gpt2")

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

# Tokenize and preprocess the dataset
def preprocess_data(example):
    inputs = tokenizer(example["ingredients"], padding=True, truncation=True, max_length=512)
    targets = tokenizer(example["directions"], padding=True, truncation=True, max_length=512)
    return {"input_ids": inputs.input_ids, "labels": targets.input_ids}

tokenized_dataset = dataset.map(preprocess_data, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="server/recipe_recommendation/t5/models/gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_dir="server/recipe_recommendation/t5/models/gpt2",
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# Train the model
trainer.train()

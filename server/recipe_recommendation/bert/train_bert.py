import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Load the BERT tokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt)

# **Corrected Model:** DistilBertForSequenceClassification is not suitable for seq2seq tasks. Use AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

dataset_train_validation = dataset['train'].train_test_split(test_size=0.2)
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']

def preprocess_data(batch):
  inputs = tokenizer(batch["ingredients"], padding="max_length", truncation=True, max_length=512)
  outputs = tokenizer(batch["directions"], padding="max_length", truncation=True, max_length=512)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["labels"] = outputs.input_ids
  return batch

dataset = dataset.filter(lambda example: example['ingredients'] is not None and len(example['ingredients']) > 0)
tokenized_dataset = dataset.map(preprocess_data, batched=True)
print("tokenized dataset: ", tokenized_dataset)

batch_size = 8  # Set batch size here

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
  output_dir="server/recipe_recommendation/t5/models/distilbert",
  num_train_epochs=2,
  learning_rate=1e-3,
  per_device_train_batch_size=batch_size,  # Ensure consistency with batch_size
  per_device_eval_batch_size=batch_size,  # Ensure consistency with batch_size
  weight_decay=0.01,
  evaluation_strategy="steps",
  eval_steps=20,
  logging_strategy="steps",
  logging_steps=100,
  save_strategy="steps",
  save_steps=200,
)

data_collator = DataCollatorForSeq2Seq(tokenizer)  # Assuming no internal batch size setting

# Create the Trainer
trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  data_collator=data_collator,
  train_dataset=tokenized_dataset['train'],
  eval_dataset=tokenized_dataset['validation'],
)

# Train the model
try:
  trainer.train()
except Exception as e:
  print(e)
import torch
from transformers import LongformerForSequenceClassification, LongformerTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model name
model_name = "allenai/longformer-base-4096"

# Load the tokenizer
tokenizer = LongformerTokenizer.from_pretrained(model_name)

# Load the pre-trained model
model = LongformerForSequenceClassification.from_pretrained(model_name)

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

dataset_train_validation = dataset['train'].train_test_split(test_size=0.2)
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))

prefix = "ingredients: "
def preprocess_data(examples):
    inputs = [prefix + text for text in examples["ingredients"]]
    model_inputs = tokenizer(inputs, max_length=None, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["directions"], max_length=None, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.filter(lambda example: example['ingredients'] is not None and len(example['ingredients']) > 0)
tokenized_datasets = dataset.map(preprocess_data, batched=True)
print("tokenized dataset: ", tokenized_datasets)

batch_size = 2
output_dir = "server/recipe_recommendation/t5/models/longformer-base-4096"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir=output_dir
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)
print(" training device: ", training_args.device)
print("Training the model...")
# Train the model
trainer.train()

# Save the model
trainer.save_model(output_dir)

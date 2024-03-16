from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoTokenizer
import torch
from transformers import T5ForConditionalGeneration, AdamW
from torch.nn import CrossEntropyLoss

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)
print(dataset)

# Splitting dataset into train and validation sets with an 80-20 split
dataset_train_validation = dataset['train'].train_test_split(test_size=0.2)

# Assign the split datasets to train and validation sets
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))

print(dataset['train'][7])

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-small")

# Define the maximum sequence length for both ingredients and directions
max_input_length = 512  
max_target_length = 512  

def preprocess_data(example):
    # Tokenize ingredients
    inputs = example['ingredients']
    inputs = [f"ingredients: {ingredient}" for ingredient in inputs]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length", return_tensors="pt")

    # Tokenize directions
    with tokenizer.as_target_tokenizer():
        targets = example['directions']
        targets = tokenizer(targets, max_length=max_target_length, truncation=True, padding="max_length", return_tensors="pt")

    model_inputs["labels"] = targets["input_ids"]
    return model_inputs

# Tokenize the dataset
tokenized_dataset = dataset.map(preprocess_data, batched=True)
print(tokenized_dataset)

# Define loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in tokenized_dataset["train"]:
        input_ids = torch.tensor(batch["input_ids"]).unsqueeze(0)
        attention_mask = torch.tensor(batch["attention_mask"]).unsqueeze(0)
        labels = torch.tensor(batch["labels"]).unsqueeze(0)

        #print("Input shape:", input_ids.shape)
        #print("Attention mask shape:", attention_mask.shape)
        #print("Labels shape:", labels.shape)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(tokenized_dataset["train"])
    print(f"Epoch {epoch+1}: Average Loss = {average_loss:.4f}")

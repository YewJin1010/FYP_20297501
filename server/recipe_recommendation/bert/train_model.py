import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import numpy as np

# Load the dataset
df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/ingredients_to_title.csv")

# Get the classes (ingredients)
classes = df.columns[1:].tolist()  # Exclude the 'title' column
num_classes = len(classes)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define the BERT model architecture for multi-label classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Preprocess the data
def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    features = df['title'].tolist()
    # Convert one-hot encoded labels to numpy array
    labels = df.drop(columns=['title']).values.astype(np.float32)
    return features, labels

# Tokenize and encode the input data
def tokenize_and_encode(features, labels):
    input_ids = []
    attention_masks = []
    for feature in features:
        encoded_dict = tokenizer.encode_plus(feature, add_special_tokens=True, max_length=64, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)  # Convert labels to tensor
    dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
    return dataset

# Train the model
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs=3):
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            input_ids, attention_masks, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_masks, labels = batch
                outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                total_val_loss += loss.item()

        # Calculate average losses
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{epochs}, Avg Train Loss: {avg_train_loss}, Avg Val Loss: {avg_val_loss}")

# Preprocess the data
features, labels = preprocess_data("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/ingredients_to_title.csv")

# Tokenize and encode the input data
dataset = tokenize_and_encode(features, labels)

# Split the dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create dataloaders for training and validation
batch_size = 32
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=batch_size)

# Set up optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Train the model
train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs)

# Step 7: Save the trained model
def save_model(model, model_directory):
    model_save_path = os.path.join(model_directory, "bert_model_2.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at {model_save_path}")

# Save the trained model
save_model(model, "C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/saved_models")


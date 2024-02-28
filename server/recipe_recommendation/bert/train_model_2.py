import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from transformers import BertModel
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig,AdamW, BertForSequenceClassification,get_linear_schedule_with_warmup
from tqdm import tqdm, trange,tnrange,tqdm_notebook
from sklearn.metrics import accuracy_score,matthews_corrcoef
import torch

BATCH_SIZE = 16
N_EPOCHS = 3

# Load your dataset
df = pd.read_csv('C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/recipe_recommendation/bert/datasets/title_ingredient.csv')

print(df['title'].unique())

labelencoder = LabelEncoder()
df['label_enc'] = labelencoder.fit_transform(df['title'])

print(df[['title', 'label_enc']].drop_duplicates(keep='first'))

df.rename(columns={'title':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)
df.rename(columns={'ingredients':'sentence'},inplace=True)

# Split dataset into training and validation sets
df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len())  # Adjust num_labels

# Tokenize input sentences
train_encodings = tokenizer(list(df_train['sentence']), truncation=True, padding=True)
val_encodings = tokenizer(list(df_valid['sentence']), truncation=True, padding=True)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(list(df_train['label']))
val_labels = torch.tensor(list(df_valid['label']))

train_dataset = (train_encodings, train_labels)
val_dataset = (val_encodings, val_labels)

# Define training parameters
batch_size = 4
learning_rate = 2e-5
num_epochs = 3

# Prepare data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            val_predictions.extend(predictions.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}')
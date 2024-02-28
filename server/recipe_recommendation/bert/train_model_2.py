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
from torch.utils.data import Dataset, DataLoader

# Create a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
# Load your dataset
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/title_ingredient.csv')

print(df['title'].unique())

label_encoder = LabelEncoder()
df['label_enc'] = label_encoder .fit_transform(df['title'])

print(df[['title', 'label_enc']].drop_duplicates(keep='first'))

df.rename(columns={'title':'label_desc'},inplace=True)
df.rename(columns={'label_enc':'label'},inplace=True)
df.rename(columns={'ingredients':'sentence'},inplace=True)

# Split dataset into training and validation sets
train_df, val_df  = train_test_split(df, test_size=0.2, random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Define training parameters
batch_size = 4
learning_rate = 2e-5
num_epochs = 3
max_length = 128

# Create dataset objects
train_dataset = CustomDataset(train_df['sentence'], train_df['label'], tokenizer, max_length)
val_dataset = CustomDataset(val_df['sentence'], val_df['label'], tokenizer, max_length)

# Create data loaders
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

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

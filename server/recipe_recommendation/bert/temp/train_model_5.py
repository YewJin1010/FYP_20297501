# Import the necessary modules
import pandas as pd
import torch, re
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load the data from a CSV file
data = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv')

label_map_file = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/label_map.txt'

# Read the label map from the text file
label_map = {}

with open(label_map_file, 'r') as f:
    for line in f:
        title, label = line.split(": ")
        label_map[title] = int(label)


# Define a custom dataset class
class RecipeDataset(Dataset):
  def __init__(self, data, tokenizer, max_length):
    self.data = data
    self.tokenizer = tokenizer
    self.max_length = max_length
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    # Get the ingredients and the label from the data
    ingredients = self.data.iloc[index]["ingredients"] 
    label = self.data.iloc[index]["title"]
    print('ingredients: ',ingredients)
    print(len(ingredients))
    print('label: ', label)
    print(len(label))

    # Encode the ingredients using the tokenizer
    encoding = self.tokenizer.encode_plus(
      ingredients,
      add_special_tokens=True,
      max_length=self.max_length,
      padding="max_length",
      truncation=True,
      return_tensors="pt"
    )

    # Convert the label to a tensor
    label = torch.tensor(label_map[label])

    # Return the encoding and the label as a dictionary
    return {
      "input_ids": encoding["input_ids"].flatten(),
      "attention_mask": encoding["attention_mask"].flatten(),
      "label": label
    }

# Initialize the tokenizer and the model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

# Define some hyperparameters
batch_size = 32
num_epochs = 5
learning_rate = 2e-5
max_length = 128

# Split the data into train and test sets
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)

# Create data loaders for train and test sets
train_dataset = RecipeDataset(train_data, tokenizer, max_length)
test_dataset = RecipeDataset(test_data, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Move the model to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Define a function to calculate the accuracy
def get_accuracy(preds, labels):
  preds = preds.argmax(dim=1)
  return (preds == labels).float().mean()

# Train the model
for epoch in range(num_epochs):
  # Set the model to training mode
  model.train()
  # Initialize the training loss and accuracy
  train_loss = 0
  train_acc = 0
  # Loop over the training batches
  for batch in train_loader:
    # Get the input ids, attention masks, and labels from the batch
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    # Get the loss and logits from the outputs
    loss = outputs.loss
    logits = outputs.logits
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # Update the training loss and accuracy
    train_loss += loss.item()
    train_acc += get_accuracy(logits, labels).item()
  # Calculate the average training loss and accuracy
  train_loss = train_loss / len(train_loader)
  train_acc = train_acc / len(train_loader)
  # Print the training results
  print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

# Evaluate the model
# Set the model to evaluation mode
model.eval()
# Initialize the test loss and accuracy
test_loss = 0
test_acc = 0
# Loop over the test batches
with torch.no_grad():
  for batch in test_loader:
    # Get the input ids, attention masks, and labels from the batch
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    # Forward pass
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    # Get the loss and logits from the outputs
    loss = outputs.loss
    logits = outputs.logits
    # Update the test loss and accuracy
    test_loss += loss.item()
    test_acc += get_accuracy(logits, labels).item()
# Calculate the average test loss and accuracy
test_loss = test_loss / len(test_loader)
test_acc = test_acc / len(test_loader)
# Print the test results
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

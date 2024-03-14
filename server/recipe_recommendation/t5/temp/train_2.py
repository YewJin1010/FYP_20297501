import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Specify the T5 model and tokenizer versions
model_name = "t5-small"
tokenizer_name = "t5-small"

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained(tokenizer_name)

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example DataFrame with "ingredients" and "title_directions" columns
data = {
    "ingredients": [
        "1 cup flour", 
        "2 eggs", 
        "1/2 cup sugar"
    ],
    "title_directions": [
        "Mix flour and sugar in a bowl", 
        "Beat eggs in a separate bowl", 
        "Combine all ingredients and mix well"
    ]
}

df = pd.read_csv("server/recipe_recommendation/t5/csv/new_data.csv", keep_default_na=False)

print(df.info())

# Tokenize "ingredients" column
ingredients_tokens = tokenizer(df["ingredients"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Tokenize "title_directions" column
title_directions_tokens = tokenizer(df["directions"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Define your tokenized inputs
inputs = {
    "input_ids": torch.cat([ingredients_tokens["input_ids"], title_directions_tokens["input_ids"]], dim=1),
    "attention_mask": torch.cat([ingredients_tokens["attention_mask"], title_directions_tokens["attention_mask"]], dim=1)
}

# Define your fine-tuning task and labels
labels = tokenizer(["Mix ingredients", "Beat eggs", "Combine ingredients and mix"], return_tensors="pt", padding=True, truncation=True)

print(labels)

# Convert labels to the format required by T5
labels["input_ids"] = labels["input_ids"][:, :-1].contiguous()
labels["attention_mask"] = labels["attention_mask"][:, :-1].contiguous()

# Define the loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fine-tuning loop
num_epochs = 3
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits

    # Compute loss
    batch_loss = loss(logits.view(-1, logits.shape[-1]), labels["input_ids"].view(-1))

    # Backward pass
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {batch_loss.item()}")

# Save the fine-tuned model
model.save_pretrained("fine_tuned_model")

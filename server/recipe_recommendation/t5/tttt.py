import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Example DataFrame with "a" and "b" columns
df = pd.read_csv('server/recipe_recommendation/t5/csv/new_data.csv')
df['a'] = df['a'].fillna('')
df = df[:10]  # Select first 10 rows for demonstration


# Tokenize "ingredients" column
ingredients_tokens = tokenizer(df["a"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Tokenize "title_directions" column for inputs
title_directions_tokens_input = tokenizer(df["b"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Tokenize "title_directions" column for labels
title_directions_tokens_label = tokenizer(df["b"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Define your tokenized inputs
inputs = {
    "input_ids": torch.cat([ingredients_tokens["input_ids"], title_directions_tokens_input["input_ids"]], dim=1),
    "attention_mask": torch.cat([ingredients_tokens["attention_mask"], title_directions_tokens_input["attention_mask"]], dim=1)
}

# Define your tokenized labels
labels = {
    "input_ids": title_directions_tokens_label["input_ids"],
    "attention_mask": title_directions_tokens_label["attention_mask"]
}

# Define the loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fine-tuning loop
num_epochs = 3
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels["input_ids"])
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

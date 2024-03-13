import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Example DataFrame with "a" and "b" columns
df = pd.read_csv('server/recipe_recommendation/t5_2/new_data.csv')
df['ingredients'] = df['ingredients'].fillna('')
df = df[:50]  # Select first 10 rows for demonstration

# Define batch size
batch_size = 2

# Tokenize "a" and "b" columns in batches
inputs = []
labels = []

for i in range(0, len(df), batch_size):
    batch_df = df.iloc[i:i+batch_size]

    # Tokenize "ingredients" column
    ingredients_tokens = tokenizer(batch_df["ingredients"].tolist(), return_tensors="pt", padding=True, truncation=True)

    # Tokenize "title_directions" column for inputs
    title_directions_tokens_input = tokenizer(batch_df["directions"].tolist(), return_tensors="pt", padding=True, truncation=True)

    # Tokenize "title_directions" column for labels
    title_directions_tokens_label = tokenizer(batch_df["directions"].tolist(), return_tensors="pt", padding=True, truncation=True)

    # Add tokenized inputs and labels to the list
    inputs.append({
        "input_ids": torch.cat([ingredients_tokens["input_ids"], title_directions_tokens_input["input_ids"]], dim=1),
        "attention_mask": torch.cat([ingredients_tokens["attention_mask"], title_directions_tokens_input["attention_mask"]], dim=1)
    })

    labels.append({
        "input_ids": title_directions_tokens_label["input_ids"],
        "attention_mask": title_directions_tokens_label["attention_mask"]
    })

# Define the loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

# Fine-tuning loop
num_epochs = 3
for epoch in range(num_epochs):
    total_loss = 0.0
    for input_batch, label_batch in zip(inputs, labels):
        # Forward pass
        outputs = model(input_ids=input_batch["input_ids"], attention_mask=input_batch["attention_mask"], labels=label_batch["input_ids"])
        logits = outputs.logits

        # Compute loss
        batch_loss = loss(logits.view(-1, logits.shape[-1]), label_batch["input_ids"].view(-1))
        total_loss += batch_loss.item()

        # Backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(inputs)}")

# Save the fine-tuned model
model_path = "server/recipe_recommendation/t5_2/fine_tuned_model" 
tokenizer_path = "server/recipe_recommendation/t5_2/fine_tuned_tokenizer"
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

# Inferencing with the fine-tuned model
input_text = "1 cup flour, 1 cup sugar, 1 egg"
input_encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
output_ids = model.generate(input_encodings["input_ids"], attention_mask=input_encodings["attention_mask"], max_length=64, num_beams=4)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


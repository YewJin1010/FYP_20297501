from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch

# Load dataset
df = pd.read_csv('server/recipe_recommendation/t5/csv/recipes_t5.csv')

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# Define batch size
batch_size = 4  # You can adjust this value based on your available memory

# Split data into batches
num_samples = len(df)
num_batches = (num_samples + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, num_samples)

    # Prepare batch input
    ingredients_batch = df['ingredients'][start_idx:end_idx]
    title_directions_batch = df['title_directions'][start_idx:end_idx]

    encoding = tokenizer(
        ingredients_batch.tolist(),
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    input_ids, attention_mask = encoding["input_ids"], encoding.attention_mask

    target_encoding = tokenizer(
        title_directions_batch.tolist(),
        padding="longest",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    labels = target_encoding.input_ids
    labels[labels == tokenizer.pad_token_id] = -100

    # Train model
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    print(f"Batch {batch_idx + 1}/{num_batches} - Loss: {loss.item()}")

# Note: You can further customize the training loop, add optimizer, scheduler, etc.
# This example demonstrates batch processing; adapt it to your specific use case.

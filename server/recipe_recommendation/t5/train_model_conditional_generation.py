import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_score import rouge_scorer

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Example DataFrame with "a" and "b" columns
df = pd.read_csv('server/recipe_recommendation/t5/dataset/new_data.csv')
df['ingredients'] = df['ingredients'].fillna('')

# Define batch size
batch_size = 8

# Tokenize ingredients
inputs = df['ingredients'].tolist()
input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Tokenize directions (labels)
targets = df['directions'].tolist()
target_encodings = tokenizer(targets, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

# Create input IDs, attention masks, and labels
input_ids = input_encodings['input_ids']
attention_mask = input_encodings['attention_mask']
labels = target_encodings['input_ids']

# Define the loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Define ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

num_epochs = 2
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(input_ids), batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]
        batch_labels = labels[i : i + batch_size]

        # Forward pass
        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(inputs)}")

    # Evaluation using ROUGE
    with torch.no_grad():
        predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        prediction_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        rouge_scores = scorer.score(prediction_texts, labels_texts)
        print(f"ROUGE-1: {rouge_scores['rouge1'].fmeasure}, ROUGE-L: {rouge_scores['rougeL'].fmeasure}")

# Save the fine-tuned model
model_path = "server/recipe_recommendation/t5/models/t5-small-conditional-generation_2" 
tokenizer_path = "server/recipe_recommendation/t5/models/t5-small-conditional-generation_2"
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)
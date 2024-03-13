import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Example DataFrame with "a" and "b" columns
df = pd.read_csv('server/recipe_recommendation/t5_2/new_data.csv')
df['ingredients'] = df['ingredients'].fillna('')
df = df[:10]  # Select first 10 rows for demonstration


# Tokenize "ingredients" column
ingredients_tokens = tokenizer(df["ingredients"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Tokenize "title_directions" column for inputs
title_directions_tokens_input = tokenizer(df["directions"].tolist(), return_tensors="pt", padding=True, truncation=True)

# Tokenize "title_directions" column for labels
title_directions_tokens_label = tokenizer(df["directions"].tolist(), return_tensors="pt", padding=True, truncation=True)

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

batch_size = 8
model_dir = "server/recipe_recommendation/t5_2/model"
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)
metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) 
                      for label in decoded_labels]
    
    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

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
model_path = "server/recipe_recommendation/t5/fine_tuned_model" 
tokenizer_path = "server/recipe_recommendation/t5/fine_tuned_tokenizer"
model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)

import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

df = pd.read_csv('server/recipe_recommendation/t5/dataset/new_data.csv')

df = df.rename(columns={ "ingredients": "source_text", "directions": "target_text"})
df = df[['source_text', 'target_text']]
df['source_text'] = df['source_text'].fillna('')

df['source_text'] = "ingredients: " + df['source_text']

print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2)

output_dir = "server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation_tm4"
# Define the training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    learning_rate=1e-3,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    output_dir = output_dir,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=test_df,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation_tm4")

# Load the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation_tm4")

# Generate predictions
text = "ingredients: 1 cup of sugar, 2 cups of flour, 3 eggs, 1 teaspoon of vanilla extract"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=512, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Directions:", generated_text)

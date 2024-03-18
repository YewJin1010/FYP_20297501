import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load the pre-trained T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

df = pd.read_csv('server/recipe_recommendation/t5/dataset/new_data.csv')

print("Ingredients data type:", type(df["ingredients"][0]))
print("Directions data type:", type(df["directions"][0]))

# Preprocess the DataFrame
df['source_text'] = "ingredients: " + df['ingredients']
df['target_text'] = df['directions']

# Drop rows with NaN values
df.dropna(inplace=True)

# Split the DataFrame into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
df.reset_index(drop=True)
print("Train set size:", len(train_df)) 
print("Train set shape:", train_df.shape)
print("Test set size:", len(test_df))
print("Test set shape:", test_df.shape)

# Tokenize datasets
def tokenize_data(data_frame): 
    tokenized_data = None
    try:
        tokenized_data = tokenizer(
            list(data_frame["source_text"]),
            text_pair=list(data_frame["target_text"]),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
    except Exception as e:
        print("Exception occurred during tokenization:")
        print(e)
        index = 0
        for index, row in data_frame.iterrows():
            try:
                tokenizer(
                    row["source_text"],
                    text_pair=row["target_text"],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            except Exception as e:
                print(f"Index {index}:")
                print("Ingredients:", row["ingredients"])
                print("Directions:", row["directions"])
                print("Error:", e)
                print()
                break
        return None
    return tokenized_data

train_data = tokenize_data(train_df)
test_data = tokenize_data(test_df)

print("Train data:", train_data)
print("Test data:", test_data)

output_dir = "server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation_tm4"
# Define training arguments
training_args = TrainingArguments(
    output_dir = output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir=output_dir
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=test_df,
)

trainer.train()

# Save the trained model
model.save_pretrained(output_dir)

# Load the fine-tuned model
model = T5ForConditionalGeneration.from_pretrained("server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation_tm4")

# Generate predictions
text = "ingredients: 1 cup of sugar, 2 cups of flour, 3 eggs, 1 teaspoon of vanilla extract"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=512, num_return_sequences=1)

# Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Directions:", generated_text)

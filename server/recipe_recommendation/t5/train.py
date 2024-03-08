from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import torch

# Load dataset
df = pd.read_csv('server/recipe_recommendation/t5/csv/recipes_t5.csv')

tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

ingredients_list = []
for index, row in df.iterrows():
    ingredients = str(row['ingredients'])
    ingredients_list.append(ingredients)

title_directions_list = []
for index, row in df.iterrows():
    title_directions = str(row['title_directions'])
    title_directions_list.append(title_directions)

encoding = tokenizer(
    ingredients_list,
    padding="longest",
    max_length= 512,
    truncation=True,
    return_tensors="pt",
    )

input_ids, attention_mask = encoding["input_ids"], encoding.attention_mask

target_encoding = tokenizer(
    title_directions_list,
    padding="longest",
    max_length= 512,
    truncation=True,
    return_tensors="pt",
    )

labels = target_encoding.input_ids
labels[labels == tokenizer.pad_token_id] = -100

loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
print(loss.item())

exit()


# Tokenize ingredients as input ids
input_ids_list = []
for index, row in df.iterrows():
    ingredients = str(row['ingredients'])
    input_ids = tokenizer(ingredients, return_tensors="pt", padding=True, truncation=True).input_ids
    input_ids_list.append(input_ids)

labels_list = []
for index, row in df.iterrows():
    title_directions = str(row['title_directions'])
    labels = tokenizer(title_directions, return_tensors="pt", padding=True, truncation=True).input_ids
    labels_list.append(labels)

loss = model(input_ids=input_ids_list, labels=labels_list).loss
print(loss.item())

# save model
model.save_pretrained("server/recipe_recommendation/t5/model")
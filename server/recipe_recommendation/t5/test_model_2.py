import torch
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk

# Load the fine-tuned T5 model and tokenizer
model_path = "server/recipe_recommendation/t5/models/t5-small-conditional-generation" 

tokenizer_path = "server/recipe_recommendation/t5/models/t5-small-conditional-generation"
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Example DataFrame with "ingredients" column
df = pd.read_csv('server/recipe_recommendation/t5/dataset/new_data.csv')
df['ingredients'] = df['ingredients'].fillna('')
df = df[:100]  

# Define batch size
batch_size = 4

with torch.no_grad():
    # Tokenize the input text
    inputs = "ingredients: " + "1 cup of flour\n2 eggs\n3 cups of sugar\n"
    input_encodings = tokenizer(inputs, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    # Generate output sequences
    output = model.generate(input_ids=input_encodings['input_ids'], 
                            attention_mask=input_encodings['attention_mask'],
                            num_beams=8, 
                            do_sample=True, 
                            min_length=10, 
                            max_length=64)

    # Decode output sequences
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
    print("Prediction: ", predicted_title)
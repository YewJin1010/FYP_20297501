import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

df = pd.read_csv('server/chatbot_bert/dataset.csv')
possible_labels = df.intent.unique()

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))

# Input sentence
input_sentence = "recommend me a recipe with chocoalte"

# Preprocess and tokenize the input
tokenized_input = tokenizer.encode_plus(input_sentence, 
                                        add_special_tokens=True, 
                                        return_attention_mask=True, 
                                        padding='max_length', 
                                        max_length=256, 
                                        return_tensors='pt')

# Make predictions
with torch.no_grad():
    outputs = model(**tokenized_input)

# Convert logits to probabilities
probs = torch.softmax(outputs.logits, dim=-1)

# Get the predicted intent label
predicted_label_idx = torch.argmax(probs, dim=-1).item()
predicted_intent = possible_labels[predicted_label_idx]

# Print predicted intent
print(f"Predicted intent: {predicted_intent}")

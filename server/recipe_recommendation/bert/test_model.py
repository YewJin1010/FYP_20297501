import torch, os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

def load_model(model_directory, num_classes):
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
    model_load_path = os.path.join(model_directory, "bert_model_2.pth")
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    return model

# Function to encode input ingredients using one-hot encoding
def encode_input_ingredients(input_ingredients, all_ingredients):
    input_encoded = [1 if ingredient in input_ingredients else 0 for ingredient in all_ingredients]
    return torch.tensor([input_encoded])

# Function for model inference
def predict_recipe_titles(model, input_encoded):
    model.eval()
    with torch.no_grad():
        outputs = model(input_encoded)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
    return probabilities

# Function to retrieve top-K recipe titles
def get_top_k_predictions(probabilities, titles, k=5):
    top_k_probabilities, top_k_indices = torch.topk(probabilities, k)
    top_k_titles = [titles[idx] for idx in top_k_indices[0]]
    return top_k_titles, top_k_probabilities[0]

# Load the dataset
df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/ingredients_to_title.csv")
# Extract the titles column from the DataFrame
titles = df['title'].tolist()

# Get the classes (ingredients)
all_ingredients = df.columns[1:].tolist()  # Exclude the 'title' column
num_classes = len(all_ingredients)

# Load the trained model
model_path = "C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/saved_models"
trained_model = load_model(model_path, num_classes)

# Example usage
input_ingredients = ['banana', 'cinnamon', 'whole wheat flour', 'white sugar', 'milk', 'vanilla', 'baking soda']
# Encode input ingredients using the same encoding scheme used during training
input_encoded = encode_input_ingredients(input_ingredients, all_ingredients)
print("Input Encoded:", input_encoded)
# Perform model inference to get probabilities for each recipe title
probabilities = predict_recipe_titles(trained_model, input_encoded)
print("Probabilities:", probabilities)
# Retrieve the top-K predicted recipe titles
# Retrieve the top-K predicted recipe titles
top_k_titles, top_k_probabilities = get_top_k_predictions(probabilities, titles, k=5)
print("Top-K titles:", top_k_titles)
print("Top-K Probabilities:", top_k_probabilities)
print("Top-K Predictions:")
for title, probability in zip(top_k_titles, top_k_probabilities):
    print(f"{title}: Probability={probability:.4f}")

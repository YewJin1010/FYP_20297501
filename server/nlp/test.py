from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import torch
import json

tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt")

# Read the recipe.json file
with open('cake_recipes.json', 'r') as file:
    recipes_data = json.load(file)

# Process each recipe
for recipe_data in recipes_data:
    # Extract directions from the recipe data
    directions = recipe_data["directions"]

    # Tokenize the recipe directions
    directions_tokens = tokenizer.encode(directions, return_tensors="pt", max_length=512, truncation=True)

    # Generate output based on user input
    user_input = "How do I make the " + recipe_data["title"] + "?"
    user_input_tokens = tokenizer.encode(user_input, return_tensors="pt")

    # Combine user input with recipe directions
    input_tokens = torch.cat([user_input_tokens, directions_tokens], dim=1)

    # Generate response using the model
    output_tokens = model.generate(input_tokens, max_length=200, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

    # Decode and print the generated response
    generated_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_response)
    print("\n" + "="*50 + "\n")  # Separate recipes for clarity
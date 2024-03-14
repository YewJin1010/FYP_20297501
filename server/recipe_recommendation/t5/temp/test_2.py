from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model_path = "server/recipe_recommendation/t5_2/fine_tuned_model"
tokenizer_path = "server/recipe_recommendation/t5_2/fine_tuned_tokenizer"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# Input
input_prompt = "milk, chocolate, banana"

# Tokenize input
input_ids = tokenizer(input_prompt, return_tensors="pt")['input_ids']

# Generate output
output = model.generate(input_ids)

# Decode generated output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Title Directions:", decoded_output)

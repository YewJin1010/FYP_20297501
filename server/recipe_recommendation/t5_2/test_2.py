import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned T5 model and tokenizer
model_path = "server/recipe_recommendation/t5_2/fine_tuned_model"
tokenizer_path = "server/recipe_recommendation/t5_2/fine_tuned_tokenizer"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

# Input prompt
input_prompt = "milk, chocolate, banana"

# Tokenize input
input_ids = tokenizer.encode(input_prompt, return_tensors="pt", padding=True, truncation=True, max_length=125)
attention_mask = input_ids.ne(tokenizer.pad_token_id)

# Generate output
output = model.generate(input_ids, attention_mask=attention_mask, do_sample=False)
print(" OUTPUT: ", output)
# Decode generated output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Title Directions:", decoded_output)

from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "server/recipe_recommendation/t5/fine_tuned_model"
tokenizer_path = "server/recipe_recommendation/t5/fine_tuned_model"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

input_prompt = "flour, eggs, milk"

input_ids = tokenizer(input_prompt, return_tensors="pt")
output = model.generate(**input_ids)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Title Directions:", decoded_output)
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "fine_tuned_model"
tokenizer_path = "fine_tuned_tokenizer"

model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

input_prompt = "flour, eggs, milk"

input_ids = tokenizer(input_prompt, return_tensors="pt")
output = model.generate(input_ids)
print("output:", output)

decoded_output = tokenizer.decode(output[1], skip_special_tokens=True)

print("Generated Title Directions:", decoded_output)
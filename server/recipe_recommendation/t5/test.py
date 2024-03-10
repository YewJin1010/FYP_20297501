from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "server/recipe_recommendation/t5/trained_models/t5small_model"
tokenizer_path = "server/recipe_recommendation/t5/trained_models/t5small_tokenizer"

# Load the saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)

input_prompt = "1 cup of flour, 2 eggs, 1 cup of milk"

input_list = [input_prompt]

# Tokenize input prompt
inputs = tokenizer(
    input_list, 
    return_tensors="pt", 
    max_length=512,
    padding="longest", 
    truncation=True
    )

# Generate text using the loaded model
output = model.generate(
    input_ids = inputs["input_ids"], 
    attention_mask = inputs["attention_mask"],
    do_sample=False
    )

print(output)
# Decode generated output
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("Output: ", decoded_output)

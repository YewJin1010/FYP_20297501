import nltk
import string
import numpy as np
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import torch

# Load model and tokenizer
model_name = "t5-small-fine-tuned_20-03-2024_19-35-26/checkpoint-3000"
model_dir = "recipe_recommendation/t5/models/" + model_name
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Constants
max_input_length = 300
max_target_length = 600

def generate_recipe(ingredients):
    # Tokenize inputs
    ingredients_prefix = "ingredients: "
    inputs = [ingredients_prefix + ingredients]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")

    # Generate recipe
    output = model.generate(**model_inputs, num_beams=8, do_sample=True, min_length=10, max_length=max_target_length)
    recipe = tokenizer.batch_decode(output, skip_special_tokens=True)
    return recipe


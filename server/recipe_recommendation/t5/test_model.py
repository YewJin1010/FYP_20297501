import nltk
import string
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset, load_metric
import torch

# Load model and tokenizer
model_name = "t5-small-fine-tuned/checkpoint-1400"
model_dir = "server/recipe_recommendation/t5/models/" + model_name
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Constants
max_input_length = 300
max_target_length = 600

# Load dataset
dataset_path = 'server/recipe_recommendation/t5/dataset/'
dataset = load_dataset(dataset_path)

# Split dataset
dataset_train_test = dataset['train'].train_test_split(test_size=100)
dataset_train_validation = dataset_train_test['train'].train_test_split(test_size=200)
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']
dataset['test'] = dataset_train_test['test']

# Preprocess
model_checkpoint = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset_cleaned = dataset.filter(
    lambda example: example['ingredients'] is not None 
                    and example['directions'] is not None
                    and len(example['ingredients']) >= 100
                    and len(example['directions']) >= 100
)

prefix = "ingredients: "

def clean_text(text):
    sentences = nltk.sent_tokenize(text.strip())
    sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
    sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                    if len(sent) > 0 and sent[-1] in string.punctuation]
    text_cleaned = "\n".join(sentences_cleaned_no_titles)
    return text_cleaned

def preprocess_data(examples):
    texts_cleaned = [clean_text(text) for text in examples["ingredients"]]
    inputs = [prefix + text for text in texts_cleaned]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["directions"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_cleaned.map(preprocess_data, batched=True)

# Testing
print("TEST 1: Ingredients + measurements")
text = dataset['test'][30]['ingredients']
print("TEXT: ", text)
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=600)
print("Output: ", output)
print("\n")

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print("Decoded output: ", decoded_output)
print("\n")

predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
print("Prediction: ", predicted_title)
print("\n")

# Test 2
print("TEST 2: no measurements + many ingredients")
#text = "flour, sugar, eggs, chocolate, apple, banana"
text = " brown sugar, raisins, water, shortening, baking soda, salt, ground cinnamon, nutmeg, cloves, flour"
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=600)
print("Output: ", output)
print("\n")

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print("Decoded output: ", decoded_output)
print("\n")

predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
print("Prediction: ", predicted_title)
print("\n")

# Test 3
print("TEST 3: no measurements + few ingredients")
#text = "flour, sugar, eggs, chocolate, apple, banana"
text = "flour, yeast, apple, banana"
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=600)
print("Output: ", output)
print("\n")

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
print("Decoded output: ", decoded_output)
print("\n")

predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
print("Prediction: ", predicted_title)
print("\n")

# Test 4
print("TEST 4")
# text = """1 cup of milk, 2 cups of sugar, 1 chocolate"""
# text = "1 banana, cream, 1 cup of cinnamon"
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]
print("Predicted text: ", predicted_title)
print("\n")
# Compute Metrics
metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# Prepare Test Dataset
test_tokenized_dataset = tokenized_datasets["test"]
def preprocess_test(examples):
    inputs = [prefix + text for text in examples["ingredients"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    return model_inputs

test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True)
test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

# Generate Predictions
all_predictions = []
for i, batch in enumerate(dataloader):
    predictions = model.generate(**batch)
    all_predictions.append(predictions)

all_predictions_flattened = [pred for preds in all_predictions for pred in preds]
all_titles = tokenizer(test_tokenized_dataset["directions"], max_length=max_target_length, truncation=True, padding="max_length")["input_ids"]
predictions_labels = [all_predictions_flattened, all_titles]

# Compute Metrics
compute_metrics(predictions_labels)

                                     

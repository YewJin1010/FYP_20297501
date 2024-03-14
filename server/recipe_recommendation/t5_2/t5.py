"""
!pip install datasets
!pip install transformers
!pip install sentencepiece
!pip install rouge_score # for model evaluation
!pip install sacrebleu
!pip install meteor
!pip install evaluate
!pip install sentence_transformers
!pip install accelerate
!pip install nltk
"""
import transformers
from datasets import load_dataset, load_metric
import pandas as pd
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

dataset_path = 'server/recipe_recommendation/t5_2/dataset/'
dataset = load_dataset(dataset_path)
print(dataset)

dataset_train_test = dataset['train'].train_test_split(test_size = 100)
dataset_train_validation = dataset_train_test['train'].train_test_split(test_size=200)

# Assign the split datasets to train, validation, and test sets
dataset['train'] = dataset_train_validation['train']
dataset['validation'] = dataset_train_validation['test']
dataset['test'] = dataset_train_test['test']

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))
print("Number of rows in the test split:", len(dataset['test']))

print(dataset['train'][0])
print(dataset)

# Preprocess
model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset_cleaned = dataset.filter(
    lambda example: example['ingredients'] is not None and example['directions'] is not None
                    and (len(example['ingredients']) >= 100)
                    and (len(example['directions']) >= 100)
)

prefix = "ingredients: "
max_input_length = 512
max_target_length = 512

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_titles)
  return text_cleaned

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["ingredients"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["directions"], max_length=max_target_length,
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_datasets = dataset_cleaned.map(preprocess_data, batched=True)
print(tokenized_datasets)

batch_size = 8
model_name = "t5-base-medium-title-generation"
model_dir = "server/recipe_recommendation/t5_2/models/" + model_name

print(model_dir)

training_args = Seq2SeqTrainingArguments(
    output_dir = model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                      for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)

    # Extract ROUGE f1 scores
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                      for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

print("Model checkpoint: ", model_checkpoint)

# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=training_args ,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

save_dir = "server/recipe_recommendation/t5_2/models/" + model_name

trainer.save_model(save_dir)
print("Saved model to:", save_dir)

model_name = "t5-small-medium-title-generation/checkpoint-200"
model_dir = "server/recipe_recommendation/t5_2/models/" + model_name

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512

# Test 1
text = dataset['test'][100]['ingredients']
print("Text: ", text)
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print("Prediction: ", predicted_title)

# Test 2
#text = """1 cup of milk, 2 cups of sugar, 1 chocolate"""
#text = "1 banana, cream, 1 cup of cinnamon"
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir '{model_dir}'/runs


import torch

# get test split
test_tokenized_dataset = tokenized_datasets["test"]

# pad texts to the same length
def preprocess_test(examples):
  inputs = [prefix + text for text in examples["ingredients"]]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True,
                           padding="max_length")
  return model_inputs

test_tokenized_dataset = test_tokenized_dataset.map(preprocess_test, batched=True)

# prepare dataloader
test_tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
dataloader = torch.utils.data.DataLoader(test_tokenized_dataset, batch_size=32)

# generate text for each batch
all_predictions = []
for i,batch in enumerate(dataloader):
  predictions = model.generate(**batch)
  all_predictions.append(predictions)

# flatten predictions
all_predictions_flattened = [pred for preds in all_predictions for pred in preds]

# tokenize and pad titles
all_titles = tokenizer(test_tokenized_dataset["directions"], max_length=max_target_length,
                       truncation=True, padding="max_length")["input_ids"]

# compute metrics
predictions_labels = [all_predictions_flattened, all_titles]
compute_metrics(predictions_labels)


import transformers
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer

dataset = load_dataset('server/recipe_recommendation/t5_2')
print(dataset)

## DO NOT RUN
# Assuming dataset is a dictionary with 'train', 'validation', and 'test' splits
for train in dataset:
    dataset[train] = dataset[train][:80]

# Split the dataset into train, validation, and test sets
train_val_test = dataset['train'].train_test_split(test_size=0.3)
train_val = train_val_test['train'].train_test_split(test_size=0.2)

# Assign the split datasets to train, validation, and test sets
dataset['train'] = train_val['train']
dataset['validation'] = train_val['test']
dataset['test'] = train_val_test['test']

print("Number of rows in the train split:", len(dataset['train']))
print("Number of rows in the validation split:", len(dataset['validation']))
print("Number of rows in the test split:", len(dataset['test']))

print(dataset['train'][0])
print(dataset)


model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Wait a while before running
dataset_cleaned = dataset.filter(
    lambda df: (len(df['ingredients']) >= 100) and
     (len(df['directions']) >= 100)
)

prefix = "ingredients: "
max_input_length = 512
max_target_length = 64

def clean_text(text):
  sentences = nltk.sent_tokenize(text.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_directions = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  text_cleaned = "\n".join(sentences_cleaned_no_directions)
  return text_cleaned

def preprocess_data(df):

  ingredients_cleaned = [clean_text(ingredients) for ingredients in df["ingredients"]]
  inputs = [prefix + ingredients for ingredients in ingredients_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(df["directions"], max_length=max_target_length,
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

tokenized_datasets = dataset_cleaned.map(preprocess_data, batched=True)
print(tokenized_datasets)




batch_size = 8
model_name = "t5-base-medium-title-generation"
model_dir = f"/content/drive/MyDrive/T5/models/{model_name}"

print(model_dir)

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False, ## RAISING ISSUE
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

metric = load_metric("rouge")

import numpy as np

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

# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

save_dir = f'/content/drive/MyDrive/T5/models/'
trainer.save_model(save_dir)
print("Saved model to:", save_dir)

model_name = "t5-base-medium-title-generation/checkpoint-200"
model_dir = f"/content/drive/MyDrive/T5/models/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512

text = """1 cup of milk, 2 cups of sugar, 1 chocolate"""
inputs = ["ingredients: " + text]

inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
predicted_title = nltk.sent_tokenize(decoded_output.strip())[0]

print(predicted_title)

df = pd.read_csv('/content/drive/MyDrive/T5/new_data.csv')

# Calculate the number of rows for training and testing
total_rows = len(df)
train_rows = int(0.8 * total_rows)
test_rows = total_rows - train_rows

# Split dataset into training and test sets without shuffling
train_df = df[:train_rows]
test_df = df[train_rows:]

print("Number of training rows: ", len(train_df))
print("Number of testiing rows: ", len(test_df))

# Preprocess
import nltk
nltk.download('punkt')
import string
from transformers import AutoTokenizer

model_checkpoint = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Filter the train_df DataFrame based on the length of 'ingredients' and 'directions' columns
train_df_cleaned = train_df[
    (train_df['ingredients'].str.len() >= 100)
    & (train_df['directions'].str.len() >= 100)]

test_df_cleaned = test_df[
    (test_df['ingredients'].str.len() >= 100)
    & (test_df['directions'].str.len() >= 100)]

# Drop NaN rows after filtering
train_df_cleaned = train_df_cleaned.dropna()
test_df_cleaned = test_df_cleaned.dropna()

print("Number of training rows: ", len(train_df_cleaned))
print("Number of testiing rows: ", len(test_df_cleaned))

prefix = "ingredients: "
max_input_length = 512
max_target_length = 512
def clean_ingredients(ingredients):
  sentences = nltk.sent_tokenize(ingredients.strip())
  sentences_cleaned = [s for sent in sentences for s in sent.split("\n")]
  sentences_cleaned_no_titles = [sent for sent in sentences_cleaned
                                 if len(sent) > 0 and
                                 sent[-1] in string.punctuation]
  ingredients_cleaned = "\n".join(sentences_cleaned_no_titles)
  return ingredients_cleaned

def preprocess_data(df):
  ingredients_cleaned = [clean_ingredients(ingredients) for ingredients in df['ingredients']]
  inputs = [prefix + ingredients for ingredients in ingredients_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(df['directions'].tolist(), max_length=max_target_length,
                       truncation=True) ### df directions str list issue
  model_inputs["labels"] = labels["input_ids"]
  return model_inputs

# Pass both train_df and test_df to the preprocess_data function
train_tokenized_datasets = preprocess_data(train_df_cleaned)
test_tokenized_datasets = preprocess_data(test_df_cleaned)

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

batch_size = 8
model_name = "t5-base-medium-title-generation"
model_dir = f"/content/drive/MyDrive/T5/{model_name}"

## ONLY RUN IF NOT VERSION 0.27.2
import accelerate

accelerate.__version__
!pip uninstall accelerate
!pip install accelerate==0.27.2
accelerate.__version__

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=200,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=False, ## Causing issues
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)

data_collator = DataCollatorForSeq2Seq(tokenizer)

!pip install rouge_score

metric = load_metric("rouge")

import numpy as np

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

# Function that returns an untrained model to be trained
def model_init():
    return AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=test_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Commented out IPython magic to ensure Python compatibility.
# Start TensorBoard before training to monitor it in progress
# %load_ext tensorboard
# %tensorboard --logdir '{model_dir}'/runs



trainer.train()

model_name = "t5-base-medium-title-generation/checkpoint-2000"
model_dir = f"drive/MyDrive/Models/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512
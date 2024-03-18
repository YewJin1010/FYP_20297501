import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split

nltk.download('punkt')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Load and preprocess the dataset
df = pd.read_csv('server/recipe_recommendation/t5/dataset/new_data.csv')
df = df.rename(columns={"ingredients": "source_text", "directions": "target_text"})
df = df[['source_text', 'target_text']]
df['source_text'] = df['source_text'].fillna('')
df['source_text'] = "ingredients: " + df['source_text']

# Split the dataset
train_df, test_df = train_test_split(df, test_size=0.2)
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# Tokenize the dataset
def tokenize_data(data_frame): 
    tokenized_data = tokenizer(
        list(data_frame["source_text"]),
        text_pair=list(data_frame["target_text"]),
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    tokenized_data = {k: v for k, v in tokenized_data.items() if k in ["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"]}
    return tokenized_data


train_data = tokenize_data(train_df)
test_data = tokenize_data(test_df)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir="server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation",
)

# Define the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model("server/recipe_recommendation/t5/models/t5-small-medium-conditional_generation")

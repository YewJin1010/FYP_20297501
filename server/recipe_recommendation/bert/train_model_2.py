import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import transformers
from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import keras

BATCH_SIZE = 16
N_EPOCHS = 3

# Load your dataset
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/title_ingredient.csv')

X = df['ingredients'].values
y = df['title'].values

# Tokenize input data
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
X_tokenized = tokenizer(X.tolist(), padding=True, truncation=True, return_tensors='tf')

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((X_tokenized, y))
# Define model
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Choose optimizer and loss function
optimizer = Adam(learning_rate=5e-5)
loss = SparseCategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

model.fit(dataset.shuffle(len(X)).batch(BATCH_SIZE), epochs=N_EPOCHS)

# Save the model
model.save('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/models/distilbert_model')
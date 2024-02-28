import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from transformers import DistilBertTokenizer, TFBertForSequenceClassification, TFDistilBertForSequenceClassification
import keras
from keras.utils import to_categorical

BATCH_SIZE = 16
N_EPOCHS = 3

# Load your dataset
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/title_ingredient.csv')

# split dataset into training and validation
df_train, df_valid = train_test_split(df, test_size=0.2)

label_map = {}

# Read the label map txt file
with open('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/label_map.txt', 'r') as file:
    lines = file.readlines()

# Create the label map dictionary
for line in lines:
    label, index = line.strip().split(': ')
    label_map[label] = int(index)

# read label map txt file
df_train['title'] = df_train['title'].map(label_map)
df_valid['title'] = df_valid['title'].map(label_map)

print(df_train.title)
print(len(df_train.title))
print(df_valid.title)
print(len(df_valid.title))
      
num_classes_train = len(df_train.title)
num_classes_valid = len(df_valid.title)

# Use one-hot encoding for the 'title' column
y_train = to_categorical(df_train['title'], num_classes=num_classes_train)
y_valid = to_categorical(df_valid['title'], num_classes=num_classes_valid)

print(y_train)


# Tokenize input data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_ingredients = [tokenizer.encode(ingredients, max_length=128, padding='max_length', truncation=True) for ingredients in df['ingredients']]

# Tokenize the titles (labels)
tokenized_titles = [tokenizer.encode(title, max_length=128, padding='max_length', truncation=True) for title in df['title']]

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((tokenized_ingredients, tokenized_titles))

# Define model
model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

num_classes = 10

# Get the output layer of the pre-trained model
output_layer = model.get_layer('classifier')

# Remove the output layer from the model
model_output = output_layer.output
model = tf.keras.Model(inputs=model.input, outputs=model_output)

# Add a new dense layer with the desired number of classes
new_output_layer = Dense(num_classes, activation='softmax', name='new_output_layer')(model_output)

# Create a new model with the modified output layer
bert_model = tf.keras.Model(inputs=model.input, outputs=new_output_layer)

# Choose optimizer and loss function
optimizer = Adam(learning_rate=5e-5)
loss = SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

# Compile the model
bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

bert_model.fit(dataset.shuffle(len(tokenized_ingredients)).batch(BATCH_SIZE), epochs=N_EPOCHS)

# Save the model
bert_model.save('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/models/distilbert_model')
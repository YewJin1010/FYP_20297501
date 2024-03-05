import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

csv = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/dataset.csv'
df = pd.read_csv(csv)

# Split the data into features and labels
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Label encode the labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# convert ingredients to numpy array
X = x.to_numpy()

# create FNN model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(y), activation='softmax'))  # Softmax for multi-class classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2)

# Save the model
model.save('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/trained_models/model.h5')

print("Model saved successfully.")

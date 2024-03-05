# Test the model
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

model = load_model('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/trained_models/model.h5') 

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

# Predict recipe title for a new set of ingredients
new_ingredients = ['butter, flour, sugar, eggs, milk, vanilla extract, baking powder, salt']
new_ingredients = np.array(new_ingredients)


predicted_title_index = np.argmax(model.predict(new_ingredients))
predicted_title = len(y)[predicted_title_index]

print(f"Predicted recipe title: {predicted_title}")
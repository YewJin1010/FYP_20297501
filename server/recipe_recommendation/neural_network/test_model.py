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

X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Label encode the labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)



# Preprocess new ingredients
def preprocess_new_ingredients(ingredients):
    # Create an empty array with the same shape as X_train (assuming X_train is your training data)
    new_ingredient_array = np.zeros((1, X_train.shape[1]))
    
    # Iterate through the new ingredients
    for ingredient in ingredients:
        if ingredient in ingredient_to_index:
            # Get the index of the ingredient
            index = ingredient_to_index[ingredient]
            # Set the corresponding column in new_ingredient_array to 1
            new_ingredient_array[0, index] = 1
            
    return new_ingredient_array

# Predict recipe title for a new set of ingredients
new_ingredients = ['butter, flour, sugar, eggs, milk, vanilla extract, baking powder, salt']
new_ingredients = new_ingredients.to_numpy()

predicted_title_index = np.argmax(model.predict(new_ingredients))
predicted_title = len(y)[predicted_title_index]

print(f"Predicted recipe title: {predicted_title}")
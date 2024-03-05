import numpy as np
import pandas as pd
from keras.models import load_model

# Recipe df
recipes_df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/recipes.csv')

# Load the trained model
model = load_model('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/trained_models/model.h5')

# Read label map
label_map_path = 'C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/label_map.txt'
label_map = {}

# class list
class_list = ['almonds', 'apple', 'apricots', 'avocado', 'baking_soda', 'banana', 'bell_pepper', 'blueberry', 'brown_sugar', 'butter', 'carrot', 'cashews', 'cheese', 'cherries', 'chestnuts', 'chickpeas', 'cinnamon', 'corn', 'dates', 'dried_currant', 'dried_figs', 'egg', 'flour', 'garlic', 'ginger', 'grapefruit', 'grapes', 'hazelnut', 'kiwi', 'lemon', 'lettuce', 'lime', 'macadamia', 'mandarin', 'mango', 'milk', 'mozzarella', 'oats', 'oil', 'onion', 'orange', 'papaya', 'paprika', 'peanuts', 'pear', 'pecan', 'pineapple', 'pistachio', 'plums', 'pomegranate', 'potato', 'pumpkin', 'raspberries', 'rice', 'rice_flour', 'salt', 'semolina', 'strawberry', 'sweet_potato', 'tomato', 'turnip', 'walnut', 'watermelon', 'wheat_flour', 'white_sugar', 'yogurt']

def read_label_map(label_map_path, label_map):
    with open(label_map_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            recipe_name = parts[0].strip()
            recipe_index = int(parts[1].strip())
            label_map[recipe_index] = recipe_name
    return label_map

def ingredient_to_vector(ingredients, class_list):
    vector = np.zeros(len(class_list))
    for ingredient in ingredients:
        if ingredient in class_list:
            vector[class_list.index(ingredient)] = 1
    return vector

def predict_recipe(new_ingredients, label_map):

    input_vector = ingredient_to_vector(new_ingredients, class_list)
    input_vector = input_vector.reshape(1, -1)  

    # Now you can pass new_input to the model for prediction
    predicted_index = np.argmax(model.predict(input_vector))

    # Map predicted index to recipe title
    predicted_recipe = label_map[predicted_index]

    return predicted_recipe

# Get ingredients from recipe
def get_ingredients(recipe_title, recipes_df):
    row_index = recipes_df[recipes_df['title'] == recipe_title].index[0]
    ingredients = recipes_df.loc[row_index, 'ingredients']
    ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]
    return ingredients_list

label_map = read_label_map(label_map_path, label_map)
ingredients = ['milk', 'banana', 'apple']
predicted_recipe = predict_recipe(ingredients, label_map)
print(f"Predicted Recipe: {predicted_recipe}")
ingredients_str = get_ingredients(predicted_recipe, recipes_df)
# line break by , 
ingredients_str = ", ".join(ingredients_str)
print(f"Ingredients: {ingredients_str}")

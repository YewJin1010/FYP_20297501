
import pandas as pd
import ast

# Read the CSV file
df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/database/cleaned_title_ingredient.csv")

def get_database_ingredients(): 
    # Extract ingredients from the column and combine into a single list
    ingredients_list = []
    for cell in df['ingredients']:
        # Remove square brackets and split the ingredients string by single quotes
        cell = cell.strip('[]')
        ingredients = [ingredient.strip() for ingredient in cell.split("''")]
        # Remove empty strings and extend the ingredients_list
        ingredients_list.extend([ingredient for ingredient in ingredients if ingredient])

    # Remove duplicates
    unique_ingredients = list(set(ingredients_list))
    return unique_ingredients


unique_ingredients = get_database_ingredients() 
# Write to notepad
with open("C:/Users/yewji/FYP_20297501/server/database/ingredients.txt", "w", encoding="utf-8") as file:
    for ingredient in unique_ingredients:
        file.write(ingredient + "\n")
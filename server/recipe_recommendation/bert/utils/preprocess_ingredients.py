# Import the necessary modules
import pandas as pd
import re
import ast

# Load the data from a CSV file
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv')
#title_ingredient
# processed_title_ingredient

# Define a function that removes the amounts and measurements from an ingredient string
def remove_amounts(ingredients):
    new_ingredients_list = []
    print('ingredients:', ingredients)

    # Define a list of words to be ignored
    ignored_words = ["egg", "eggs", "salt", "pepper", "sugar"] 

   # Define a regular expression pattern to match the amounts and measurements
    amount_pattern = r"\b(?:\w+\s+)?\d*\s*\d+\/?\d*\s*[a-zA-Z]*\s*(?!(?:\b(?:{}))\b)".format("|".join(ignored_words))

    # Define a regular expression pattern to match text inside parentheses
    bracket_pattern = r"\([^()]*\)"
    
    for ingredient in ingredients:
        print('original ingredient:', ingredient)
        # Remove the amounts and measurements
        ingredient = re.sub(amount_pattern, "", ingredient)
        # Remove text inside parentheses
        ingredient = re.sub(bracket_pattern, "", ingredient)
        print('processed ingredient:', ingredient)
        new_ingredients_list.append(ingredient)
        print("new_ingredients_list:", new_ingredients_list)
    
    return new_ingredients_list

def replace_or(df): 
    # Iterate over each cell in the "ingredients" column
    for index, row in df.iterrows():
        ingredients = row['ingredients']
        ingredients = ingredients.replace(" or ", "', '")
        df.at[index, 'ingredients'] = ingredients

"""
df['ingredients'] = df['ingredients'].apply(lambda x: ast.literal_eval(x))
# Apply the function to the 'ingredient' column using the apply() method
df['ingredient'] = df['ingredients'].apply(lambda x: remove_amounts(x))

# Drop the 'ingredients' column
df.drop(columns=['ingredients'], inplace=True)
# Rename the 'ingredient' column to 'ingredients'
df.rename(columns={'ingredient': 'ingredients'}, inplace=True)
"""

replace_or(df)
# Write the modified DataFrame to the csv file
df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv', index=False)
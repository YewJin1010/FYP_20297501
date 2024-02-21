# Import the necessary modules
import pandas as pd
import re
import ast
# Load the data from a CSV file
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/title_ingredient.csv')

# Define a function that removes the amounts and measurements from an ingredient string
def remove_amounts(ingredients):
    new_ingredients_list = []
    print('ingredients:', ingredients)
    # Define a regular expression pattern to match the amounts and measurements
    pattern = r"^\d+\/?\d*\s*[a-zA-Z]*\s*"

    for ingredient in ingredients:
        print('original ingredient:', ingredient)
        # Replace the matched text with nothing
        ingredient = re.sub(pattern, "", ingredient)
        print('processed ingredient:', ingredient)
        new_ingredients_list.append(ingredient)
        print("new_ingredients_list:", new_ingredients_list)
    
    return new_ingredients_list

df['ingredients'] = df['ingredients'].apply(lambda x: ast.literal_eval(x))
# Apply the function to the 'ingredient' column using the apply() method
df['ingredient'] = df['ingredients'].apply(lambda x: remove_amounts(x))

# Write the modified DataFrame to the csv file
df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/title_ingredient.csv', index=False)
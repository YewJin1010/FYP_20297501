
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

"""
unique_ingredients = get_database_ingredients() 
# Write to notepad
with open("C:/Users/yewji/FYP_20297501/server/database/ingredients.txt", "w", encoding="utf-8") as file:
    for ingredient in unique_ingredients:
        file.write(ingredient + "\n")
        """

def extract_ingredients_from_text(text):
    # Get the list of unique ingredients from the database
    unique_ingredients = get_database_ingredients()
    found_ingredients = []

    # Split the string into words and remove punctuation
    words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split()

    # Iterate over each ingredient
    for ingredient in unique_ingredients:
        # Check if any word from the text contains the ingredient as a substring
        if ingredient.lower() in word.lower() for word in words for ingredient_part in ingredient.split():
            found_ingredients.append(ingredient)
            print("Found ingredient: ", ingredient)
    
    print("Final found ingredients: ", found_ingredients)
    return found_ingredients

string = "can you suggest me a recipe using chocolate and milk"
ingre = extract_ingredients_from_text(string)
print(ingre)
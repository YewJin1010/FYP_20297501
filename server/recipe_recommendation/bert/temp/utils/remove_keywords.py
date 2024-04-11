# Import the necessary modules
import pandas as pd
import re
import ast

# Load the data from a CSV file
df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv')

"""
# Enter word to remove
word = input("Enter word to remove: ")

# Define a function to remove the specified word from each ingredient string
def remove_word_from_ingredients(ingredient, word):
    # Check if the word exists in the ingredient string
    if word in ingredient:
        # Remove the word from the ingredient string
        ingredient = ingredient.replace(word, "")
    return ingredient

# Apply the function to the 'ingredients' column using the apply() method
df['ingredients'] = df['ingredients'].apply(lambda x: remove_word_from_ingredients(x, word))
"""
# Read notepad for words to remove
with open("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/words_to_remove.txt", "r") as file:
    words_to_remove = file.readlines()

# Define a function to remove the specified word from each ingredient string
def remove_words_from_ingredients(ingredient, words_to_remove):
    # Iterate over each word to remove
    for word in words_to_remove:
        # Check if the word exists in the ingredient string
        if word.strip() in ingredient:
            # Remove the word from the ingredient string
            ingredient = ingredient.replace(word.strip(), "")
    return ingredient

# Apply the function to the 'ingredients' column using the apply() method
df['ingredients'] = df['ingredients'].apply(lambda x: remove_words_from_ingredients(x, words_to_remove))

# Write the modified DataFrame to the csv file
df.to_csv('C:/Users/yewji/FYP_20297501/server/recipe_recommendation/bert/datasets/cleaned_title_ingredient.csv', index=False)

print("done")
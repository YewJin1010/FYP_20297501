import pandas as pd
import json 
from collections import Counter
import ast
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import unidecode

nltk.download('wordnet')
nltk.download('stopwords')

def create_csv(): 
    # Assuming your JSON data is stored in a file named 'recipes.json'
    with open('C:/Users/yewji/FYP_20297501/server/object_detection/cakes/recipes.json', 'r') as json_file:
        recipes_data = json.load(json_file)

    # Convert the JSON data to a DataFrame
    df = pd.json_normalize(recipes_data)

    # Add a new column 'RecipeId'
    df.insert(0, 'id', range(1, len(df) + 1))

    # Save the DataFrame to a CSV file
    df.to_csv('cake_recipes.csv', index=False)

def word_freq():
    # Read CSV
    recipe_df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_matching/recipes.csv')
    # Initialize a vocabulary frequency distribution
    vocabulary = Counter()
    # Iterate through the 'ingredients' column in the DataFrame
    for ingredients in recipe_df['ingredients']:
        # Split the ingredients into a list
        ingredients = ingredients.split()
        # Update the vocabulary frequency distribution
        vocabulary.update(ingredients)

    # Display the top 200 words and their frequencies
    for word, frequency in vocabulary.most_common(200):
        print(f'{word};{frequency}')

def word_freq_parsed():
    # Read CSV with parsed ingredients
    parsed_recipe_df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_matching/parsed_recipe_data.csv')
    
    # Initialize a vocabulary frequency distribution
    vocabulary = Counter()

    # Iterate through the 'parsed_ingredients' column in the DataFrame
    for parsed_ingredients in parsed_recipe_df['parsed_ingredients']:
        # Split the parsed ingredients into a list
        ingredients_list = parsed_ingredients.split(', ')
        # Update the vocabulary frequency distribution
        vocabulary.update(ingredients_list)

    # Display the top 200 words and their frequencies
    for word, frequency in vocabulary.most_common(200):
        print(f'{word};{frequency}')

def ingredient_parser(ingredients):
    # measures and common words (already lemmatized)   
    measures = ['teaspoon', 't', 'tsp.', 'tablespoon', 'T', 'cup', 'ounce', 'oz.', 'pound', 'lb.', 'quart', 'qt.', 'gallon', 'gal.', 'pint', 'pt.', 'pinch', 'dash', 'clove', 'can', 'jar', 'package', 'bottle', 'bunch', 'sprig', 'small', 'medium', 'large', 'extra-large', 'heaped', 'level', 'even', 'rounded', 'scant', 'generous', 'handful', 'inch']
    words_to_remove = ['1', 'temperature', 'room', 'stick','fresh', 'a', 'red', 'bunch', 'slice', 'sliced', 'diced', 'chopped', 'minced', 'grated', 'peeled', 'cored', 'seeded', 'halved', 'quartered', 'cubed', 'crushed', 'mashed', 'pitted', 'shredded', 'whole', 'fresh', 'frozen', 'cooked', 'drained', 'rinsed', 'dried', 'canned', 'thawed', 'softened', 'packed', 'lightly', 'firmly', 'divided', 'plus', 'more', 'to', 'taste', 'as', 'needed', 'optional', 'for']

    # We first get rid of all the punctuation
    translator = str.maketrans('', '', string.punctuation)
    
    # initialize nltk's lemmatizer    
    lemmatizer = WordNetLemmatizer()

    ingred_list = []

    for i in ingredients:
        i = i.translate(translator)
        # We split up with hyphens as well as spaces
        items = re.split(' |-', i)
        # Get rid of words containing non-alphabet letters
        items = [word for word in items if word.isalpha()]
        # Turn everything to lowercase
        items = [word.lower() for word in items]
        # remove accents
        items = [unidecode.unidecode(word) for word in items]
        # Lemmatize words so we can compare words to measuring words
        items = [lemmatizer.lemmatize(word) for word in items]
        # get rid of stop words
        stop_words = set(stopwords.words('english'))
        items = [word for word in items if word not in stop_words]
        # Gets rid of measuring words/phrases, e.g. heaped teaspoon
        items = [word for word in items if word not in measures]
        # Get rid of common easy words
        items = [word for word in items if word not in words_to_remove]
        if items:
            ingred_list.append(', '.join(items))

    # Save the DataFrame to a new CSV file with the parsed ingredients
    parsed_recipe_df = pd.DataFrame({'parsed_ingredients': ingred_list})
    parsed_recipe_df.to_csv('parsed_recipe_data.csv', index=False, header=True)  

    return ingred_list

print("1. Create csv")
print("2. Word frequency")
print("3. Word frequency Parse ingredients")
print("4. Parse ingredients")
selection = int(input())
if selection == 1:
    create_csv()

elif selection == 2:
    word_freq()

elif selection == 3:
    word_freq_parsed()

elif selection == 4:
    # Read the recipes data from 'recipe.csv'
    recipe_df = pd.read_csv('C:/Users/yewji/FYP_20297501/server/recipe_matching/recipes.csv')

    # Call the ingredient_parser function with the 'ingredients' column
    parsed_ingredient_list = ingredient_parser(recipe_df['ingredients'])
    print(parsed_ingredient_list)
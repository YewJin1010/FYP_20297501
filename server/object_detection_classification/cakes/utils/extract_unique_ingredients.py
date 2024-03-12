import json
from collections import Counter

def extract_unique_ingredients():
    json_file_path = 'server/object_detection_classification/cakes/ingredients_list.json'
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Flatten the list of ingredients
    all_ingredients = [ingredient for recipe in data for ingredient in recipe.get('ingredients', [])]

    # Use Counter to count the occurrences of each ingredient
    ingredient_counter = Counter(all_ingredients)

    # Sort the unique ingredients by frequency (most to least)
    sorted_unique_ingredients = sorted(ingredient_counter.items(), key=lambda x: x[1], reverse=True)

    unique_ingredients_file_path = 'server/object_detection_classification/cakes/unique_ingredients.json'
    # Write sorted unique ingredients with counts to a new JSON file
    with open(unique_ingredients_file_path, 'w') as file:
        json.dump(sorted_unique_ingredients, file, indent=2)

extract_unique_ingredients()
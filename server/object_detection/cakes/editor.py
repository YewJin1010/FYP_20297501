import json
import webbrowser
from collections import Counter
import os

# Load the data from baking.json
with open('recipes.json', 'r') as json_file:
    data = json.load(json_file)

def extract_data_to_file(criteria):
    item_list = []

    # Iterate through each recipe
    for recipe in data:
        # Extract the specified criteria from each recipe
        item_list.append({criteria: recipe[criteria]})

    # Create a file name based on the provided criteria
    file_name = f'{criteria}_list.json'

    # Write the extracted data to the file
    with open(file_name, 'w') as output_file:
        json.dump(item_list, output_file, indent=4)
    
    print(f'Data extracted and saved to {file_name}')

def open_url_by_range(start_index, end_index):
    # Open URLs within the specified range
    for index in range(start_index, min(end_index + 1, len(data))):
        url = data[index]['url']
        webbrowser.open(url)

    print(f"Opened URLs from index {start_index} to {min(end_index, len(data) - 1)}")


def check_number_of_recipes():
    print(f'There are {len(data)} recipes in the file')

def extract_unique_ingredients():
    # Load the JSON file
    with open('ingredients_list.json', 'r') as file:
        data = json.load(file)

    # Flatten the list of ingredients
    all_ingredients = [ingredient for recipe in data for ingredient in recipe.get('ingredients', [])]

    # Use Counter to count the occurrences of each ingredient
    ingredient_counter = Counter(all_ingredients)

    # Sort the unique ingredients by frequency (most to least)
    sorted_unique_ingredients = sorted(ingredient_counter.items(), key=lambda x: x[1], reverse=True)

    # Write sorted unique ingredients with counts to a new JSON file
    with open('unique_ingredients.json', 'w') as file:
        json.dump(sorted_unique_ingredients, file, indent=2)

def remove_recipes_by_criteria_n_keyword(criteria, keyword):
    # Create a new list to store recipes that don't contain the specified keyword
    filtered_recipes = []

    keyword_lower = keyword.lower()

    # Iterate through each recipe
    for recipe in data:
        title_lower = recipe.get("title", "").lower()

        if keyword_lower not in title_lower:
            
            # Add the recipe to the filtered list
            filtered_recipes.append(recipe)

    return filtered_recipes

def remove_recipes_by_ingredient(data, ingredient_to_remove):
    # Convert the ingredient to lowercase for case-insensitive comparison
    ingredient_lower = ingredient_to_remove.lower()

    # Create a new list to store recipes that don't contain the specified ingredient
    filtered_recipes = []

    # Iterate through each recipe
    for recipe in data:
        # Convert the ingredients to lowercase for case-insensitive comparison
        ingredients_lower = [ingredient.lower() for ingredient in recipe.get("ingredients", [])]
        
        # Check if the lowercase ingredient is not in the lowercase list of 'ingredients' of the recipe
        if all(ingredient_lower not in ingredient for ingredient in ingredients_lower):
            # Add the recipe to the filtered list
            filtered_recipes.append(recipe)

    return filtered_recipes

def update_ingredient_list():
    # Load existing data or initialize empty lists
    try:
        with open('ingredient_lists.json', 'r') as file:
            data = json.load(file)
            primary_ingredients = data.get('primary', [])
            secondary_ingredients = data.get('secondary', [])
    except FileNotFoundError:
        primary_ingredients = []
        secondary_ingredients = []

    while True:
        print("Enter 1 to add a primary ingredient,\n2 to add a secondary ingredient, \nor 'done' to finish:")
        user_input = input()

        if user_input.lower() == 'done':
            break

        if user_input == '1':
            ingredient = input("Enter the primary ingredient name: ")
            primary_ingredients.append(ingredient)
        elif user_input == '2':
            ingredient = input("Enter the secondary ingredient name: ")
            secondary_ingredients.append(ingredient)
        else:
            print("Invalid input. Please enter 1, 2, or 'done'.")

    # Save the updated lists to a JSON file
    with open('ingredient_lists.json', 'w') as file:
        json.dump({'primary': primary_ingredients, 'secondary': secondary_ingredients}, file, indent=2)

def remove_unique_ingredient_entries():
    # Load the unique ingredients data
    with open('unique_ingredients.json', 'r') as file:
        unique_ingredients = json.load(file)

    # Take user input for words to remove
    words_to_remove = input("Enter words (comma-separated) to remove from unique ingredients: ").split(',')

    # Remove entries containing the specified words
    filtered_ingredients = [entry for entry in unique_ingredients if not any(word.strip().lower() in entry[0].lower() for word in words_to_remove)]

    # Remove the old unique_ingredients.json file
    os.remove('unique_ingredients.json')

    # Save the filtered ingredients to a new JSON file
    with open('unique_ingredients.json', 'w') as file:
        json.dump(filtered_ingredients, file, indent=2)

    print(f"{len(filtered_ingredients)} ingredients saved to unique_ingredients.json")

def sort_and_save_ingredients():
    # Load the JSON file
    with open('ingredients_sorting.json', 'r') as file:
        data = json.load(file)

    # Sort ingredients alphabetically
    data['primary'] = sorted(data['primary'])
    data['secondary'] = sorted(data['secondary'])
   #data['tertiary'] = sorted(data['tertiary'])

    # Remove the old unique_ingredients.json file
    os.remove('ingredients_sorting.json')

    # Write sorted unique ingredients with counts to a new JSON file
    with open('ingredients_sorting.json', 'w') as file:
        json.dump(data, file, indent=2)

# User menu
print("1. Extract data to file")
print("2. Open URLs by range")
print("3. Check number of recipes")
print("4. Extract unique ingredients")
print("5. Remove recipes by keyword")
print("6. Remove recipes by ingredient")
print("7. Update ingredient list")
print("8. Remove unique ingredient entries")
print("9. Sort and save ingredients")
# User selection
selection = input("Enter selection: ")

if selection == "1":
    # User input for the criteria
    criteria = input("Enter criteria: ")

    # Call the function to extract data based on the provided criteria
    extract_data_to_file(criteria)

elif selection == "2":
    # User input for the range
    start_index = int(input("Enter start index: "))
    end_index = int(input("Enter end index: "))

    # Call the function to open URLs based on the provided range
    open_url_by_range(start_index, end_index)

elif selection == "3":	
    # Call the function to check the number of recipes
    check_number_of_recipes()

elif selection == "4":
    # Call the function to extract unique ingredients
    extract_unique_ingredients()
   
elif selection == "5":
    # User input for the criteria
    criteria = input("Enter criteria: ")
    # User input for the keyword
    keyword = input("Enter keyword: ")

    # Call the function to remove recipes based on the provided keyword
    filtered_recipes = remove_recipes_by_criteria_n_keyword(criteria, keyword)

    os.remove('recipes.json')

    # Save the filtered recipes to a new file
    with open('recipes.json', 'w') as output_file:
        json.dump(filtered_recipes, output_file, indent=4)

    print(f'{len(filtered_recipes)} recipes saved to recipes.json')
 

elif selection == "6":
    # User input for the ingredient to remove
    ingredient_to_remove = input("Enter ingredient to remove: ")

    # Call the function to remove recipes based on the provided ingredient
    filtered_recipes = remove_recipes_by_ingredient(data, ingredient_to_remove)

    os.remove('recipes.json')

    # Save the filtered recipes to a new file
    with open('recipes.json', 'w') as output_file:
        json.dump(filtered_recipes, output_file, indent=4)

    print(f'{len(filtered_recipes)} recipes saved to recipes.json')

elif selection == "7":
    # Call the function to update the ingredient list
    update_ingredient_list()

elif selection == "8": 
    # Call the function to remove unique ingredient entries
    remove_unique_ingredient_entries()

elif selection == "9":
    # Call the function to sort and save ingredients
    sort_and_save_ingredients()
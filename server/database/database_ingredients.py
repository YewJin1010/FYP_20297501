
import pandas as pd

# Read the CSV file
def get_database_ingredients(): 
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/database/title_ingredient_dataset.csv")

    # Extract ingredients from the column and combine into a single list
    ingredients_list = []
    for cell in df['ingredients']:
        # Remove square brackets and single quotes
        cell = cell.strip("[]")
        # Split the cell string by commas and remove any leading or trailing whitespace
        ingredients = [ingredient.strip() for ingredient in cell.split(",")]
        # Remove empty strings and extend the ingredients_list
        ingredients_list.extend([ingredient for ingredient in ingredients if ingredient])
    # Remove duplicates
    unique_ingredients = list(set(ingredients_list))
    return unique_ingredients

def get_sorted_ingredients():
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/database/sorted_ingredients.csv")

    ingredients = []
    for col in df.columns:
        for ingredient in df[col]:
            ingredients.append(ingredient)
    
    # drop nan
    ingredients = [x for x in ingredients if str(x) != 'nan']
    # Replace _ with space
    ingredients = [x.replace("_", " ") for x in ingredients]
    return ingredients
      
unique_ingredients = get_database_ingredients() 
# Write to notepad
with open("C:/Users/yewji/FYP_20297501/server/database/ingredients.txt", "w", encoding="utf-8") as file:
    for ingredient in unique_ingredients:
        file.write(ingredient + "\n")

def extract_ingredients_from_text(text):
    # Get the list of unique ingredients from the database
    unique_ingredients = get_sorted_ingredients()
    found_ingredients = []

    # Split the string into words and remove punctuation
    words = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split()
    print("Words: ", words)

    for ingredient in unique_ingredients:
        # Check if any word from the ingredient list is a substring of any word in the user text
        if any(ingredient.lower() in word.lower() for word in words):
            found_ingredients.append(ingredient)

    return found_ingredients

string = "can you suggest me a recipe using chocolate and milk"
ingre = extract_ingredients_from_text(string)
print(ingre)

import pandas as pd

# List of ingredients to check for
ingredients_to_check = ['almonds', 'apple', 'apricots', 'avocado', 'baking_soda', 'banana', 'bell_pepper', 'blueberry', 'brown_sugar', 'butter', 'carrot', 'cashews', 'cheese', 'cherries', 'chestnuts', 'chickpeas', 'cinnamon', 'corn', 'dates', 'dried_currant', 'dried_figs', 'egg', 'flour', 'garlic', 'ginger', 'grapefruit', 'grapes', 'hazelnut', 'kiwi', 'lemon', 'lettuce', 'lime', 'macadamia', 'mandarin', 'mango', 'milk', 'mozzarella', 'oats', 'oil', 'onion', 'orange', 'papaya', 'paprika', 'peanuts', 'pear', 'pecan', 'pineapple', 'pistachio', 'plums', 'pomegranate', 'potato', 'pumpkin', 'raspberries', 'rice', 'rice_flour', 'salt', 'semolina', 'strawberry', 'sweet_potato', 'tomato', 'turnip', 'walnut', 'watermelon', 'wheat_flour', 'white_sugar', 'yogurt']

# Function to check if each ingredient exists at least once in the DataFrame row by row
# Function to check if any word from the list exists in the DataFrame
def check_words_in_dataframe(df):
    # Flatten the DataFrame into a single list of words
    all_words = df.values.flatten().tolist()
    all_words_lower = [str(word).lower() for word in all_words]  # Convert to lowercase for case-insensitive matching
    
    # Check if any word from the list exists in the flattened list of words
    missing_words = set(ingredients_to_check) - set(all_words_lower)
    if missing_words:
        print("Words not found in the entire DataFrame:")
        for word in missing_words:
            print(word)
    else:
        print("All words found in the entire DataFrame.")

df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/sorted_ingredients.csv")
check_words_in_dataframe(df)

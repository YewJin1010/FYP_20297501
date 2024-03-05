import pandas as pd
import os

def create_csv(csv_directory, class_list):
    # Replace _ with space
    class_list = [class_.replace('_', ' ') for class_ in class_list]

    # Create a DataFrame with the class list as columns
    df = pd.DataFrame(columns=class_list) 
    # Write to CSV file
    file_path = os.path.join(csv_directory, 'dataset.csv')
    df.to_csv(file_path, index=False)

def add_titles_column(csv_directory, df, recipes_df):
    # Get the titles of the recipes
    titles = recipes_df['title'].tolist()
    # Create a DataFrame with a single column containing the titles
    titles_df = pd.DataFrame({'title': titles})
    # Concatenate the DataFrame containing the titles with the original DataFrame
    df = pd.concat([titles_df, df])

    csv_save_path = os.path.join(csv_directory, 'dataset.csv')
    df.to_csv(csv_save_path, index=False)

def one_hot_encoding(csv_directory, df, recipes_df, classes): 

    # Get the titles of the recipes
    titles_df = df['title'].tolist()
    titles_recipes_df = recipes_df['title'].tolist()
    # Initialize a list to store the one-hot encoded data
    one_hot_data = []
    for title_df in titles_df:
        if title_df in titles_recipes_df:
            row_index = titles_recipes_df.index(title_df)
            ingredients = recipes_df.loc[row_index, 'ingredients']
            # Initialize a list to store the one-hot encoding for the current recipe
            one_hot_encoding_list = []
            # Split the ingredients string into individual ingredients
            ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]
            # Check each class
            for class_ in classes:
                # Check if any part of the class exists in the ingredients
                if any(class_ in ingredient for ingredient in ingredients_list):
                    one_hot_encoding_list.append(1)
                else:
                    one_hot_encoding_list.append(0)
            one_hot_data.append([title_df] + one_hot_encoding_list)
        else:
            print(f"Title: {title_df}, Ingredients: Not found")

    # Convert the one-hot encoded data to a DataFrame
    one_hot_df = pd.DataFrame(one_hot_data, columns=['title'] + classes)

    # Save the DataFrame to CSV
    csv_save_path = os.path.join(csv_directory, 'dataset.csv')
    one_hot_df.to_csv(csv_save_path, index=False)
    
csv_directory = "C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv"
class_list = ['almonds', 'apple', 'apricots', 'avocado', 'baking_soda', 'banana', 'bell_pepper', 'blueberry', 'brown_sugar', 'butter', 'carrot', 'cashews', 'cheese', 'cherries', 'chestnuts', 'chickpeas', 'cinnamon', 'corn', 'dates', 'dried_currant', 'dried_figs', 'egg', 'flour', 'garlic', 'ginger', 'grapefruit', 'grapes', 'hazelnut', 'kiwi', 'lemon', 'lettuce', 'lime', 'macadamia', 'mandarin', 'mango', 'milk', 'mozzarella', 'oats', 'oil', 'onion', 'orange', 'papaya', 'paprika', 'peanuts', 'pear', 'pecan', 'pineapple', 'pistachio', 'plums', 'pomegranate', 'potato', 'pumpkin', 'raspberries', 'rice', 'rice_flour', 'salt', 'semolina', 'strawberry', 'sweet_potato', 'tomato', 'turnip', 'walnut', 'watermelon', 'wheat_flour', 'white_sugar', 'yogurt']

print("1. Create new dataset.csv")
print("2. Add titles column")
print("3. One-hot encoding")
choice = int(input("Select an option: "))
if choice == 1:
    create_csv(csv_directory, class_list)

elif choice == 2:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/dataset.csv")
    recipes_df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/recipes.csv")
    add_titles_column(csv_directory, df, recipes_df)

elif choice == 3:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/dataset.csv")
    recipes_df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_recommendation/neural_network/csv/recipes.csv")
    one_hot_encoding(csv_directory, df, recipes_df, class_list)
else: 
    print("Invalid input.")

import pandas as pd
import os

def get_classes(df):
    df.keys().values.tolist()
    classes = [col for col in df if col != 'title']
    return classes

def create_csv(csv_directory, df):
    # Create an empty list to store all values
    all_values = []

    # Iterate over each column after skipping the first row
    for col in df.columns:
        # Get the values of the current column
        column_values = df[col].tolist()
        # Extend the list of all values with the values of the current column
        all_values.extend(column_values)

    # Remove any empty strings from the list of all values
    all_values = [value for value in all_values if pd.notna(value)]
    all_values = [value.replace('_', ' ') if isinstance(value, str) else value for value in all_values]

    # Create a DataFrame with a single row containing all values
    single_row_df = pd.DataFrame([all_values])

    csv_save_path = os.path.join(csv_directory, 'ingredients_to_title.csv')
    single_row_df.to_csv(csv_save_path, index=False, header=False)

def add_titles_column(csv_directory, df, recipes_df):
    # Get the titles of the recipes
    titles = recipes_df['title'].tolist()
    # Create a DataFrame with a single column containing the titles
    titles_df = pd.DataFrame({'title': titles})
    # Concatenate the DataFrame containing the titles with the original DataFrame
    df = pd.concat([titles_df, df], axis=1)  # Concatenate along columns
    csv_save_path = os.path.join(csv_directory, 'ingredients_to_title.csv')
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
    csv_save_path = os.path.join(csv_directory, 'ingredients_to_title.csv')
    one_hot_df.to_csv(csv_save_path, index=False)
    
csv_directory = "C:/Users/yewji/FYP_20297501/server/recipe_matching/bert"

print("1. Create new ingredients_to_title.csv")
print("2. Add titles column")
print("3. One-hot encoding")
choice = int(input("Select an option: "))
if choice == 1:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/sorted_ingredients.csv")
    create_csv(csv_directory, df)

elif choice == 2:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/ingredients_to_title.csv")
    recipes_df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/recipes.csv")
    add_titles_column(csv_directory, df, recipes_df)

elif choice == 3:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/ingredients_to_title.csv")
    recipes_df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/recipes.csv")
    classes = get_classes(df)
    one_hot_encoding(csv_directory, df, recipes_df, classes)
else: 
    print("Invalid input.")

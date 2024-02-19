import pandas as pd
import numpy as np
import os, re, csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

recipes = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/recipes.csv")
recipes = recipes.drop('image', axis=1)

nltk_data_path = 'C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/nltk_data'
nltk.data.path.append(nltk_data_path)
# Download NLTK resources
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

def create_ingredients_csv(csv_directory): 
    df = pd.DataFrame({'ingredients': recipes['ingredients']})
     # Remove instances of ['' from each cell and replace with an empty string
    df['ingredients'] = df['ingredients'].str.replace("\[''\]", '', regex=True)
    # Convert the remaining lists to strings separated by ','
    df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(eval(x)) if pd.notna(x) else '')
    df = df['ingredients'].apply(lambda x: pd.Series(x.split(', ')))
    csv_save_path = os.path.join(csv_directory, 'ingredients.csv')
    df.to_csv(csv_save_path, index=False)

def preprocess_dataset(df, csv_directory): 
    measurements = ['cups', 'cup', 'teaspoons', 'teaspoon', 'pints', 'pint', 'large', 'small', 'ounces', 'ounce', 'pounds', 'pound', 'tablespoons', 'tablespoon', 'tbsp']
    actions = ['seperated', 'chopped', 'sliced', 'diced', 'minced', 'grated', 'peeled', 'cut', 'crushed', 'grinded', 'mixed', 'blend', 'stir', 'whisk', 'knead', 'roll', 'fold', 'beat', 'bake', 'boil', 'broil', 'fry', 'grill', 'roast', 'saute', 'simmer', 'steam', 'stew', 'braise', 'poach', 'marinate', 'season', 'baste', 'coat', 'glaze', 'garnish', 'stuff', 'tenderize', 'caramelize', 'deglaze', 'reduce', 'scald', 'sear', 'toast', 'baste', 'blanch', 'braise', 'clarify', 'cure', 'flambe', 'glaze', 'parbo', 'grated']
    # items = ['pans', 'pan', 'bowls', 'bowl', 'skillets', 'skillet', 'saucepans', 'saucepan', 'sheets', 'sheet', 'trays', 'tray', 'dishes', 'dish', 'sticks', 'stick']
    numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    others = ['extra', 'at', 'room', 'temperature']
    stopwords = measurements + actions + numbers + others
    # Iterate through all columns
    for col in df.columns:
        # Apply the replacement operation to each cell in the column
        df[col] = df[col].astype(str).apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stopwords]))
        # Remove numbers and symbols such as ()
        df[col] = df[col].apply(lambda x: re.sub(r'[^a-zA-Z\s\']', '', x))
    
    csv_save_path = os.path.join(csv_directory, 'preprocessed_ingredients.csv')
    df.to_csv(csv_save_path, index=False)

def remove_duplicates(df, csv_directory):
    # Flatten the DataFrame into a single column
    flattened_df = pd.DataFrame({'Flattened': df.stack().astype(str).values})

    # Remove duplicate cells
    unique_flattened_df = flattened_df.drop_duplicates()

    # Save the final DataFrame to the same CSV file
    csv_save_path = os.path.join(csv_directory, 'unique_ingredients.csv')
    unique_flattened_df.to_csv(csv_save_path, index=False, header=False)

def update_df(df, csv_directory):
    # Flatten the DataFrame into a single column
    flattened_df = pd.DataFrame({'Flattened': df.stack().astype(str).values})

    # Remove leading whitespace from each cell in the 'Flattened' column
    flattened_df['Flattened'] = flattened_df['Flattened'].str.lstrip()

    # Remove duplicate cells
    unique_flattened_df = flattened_df.drop_duplicates()

    # Reshape the flattened DataFrame back to its original shape
    reshaped_df = unique_flattened_df['Flattened'].str.split(',', expand=True).stack().unstack().reset_index(drop=True)

    # Remove empty cells and pad rows with empty strings
    reshaped_df = reshaped_df.apply(lambda row: ', '.join(row.dropna()), axis=1)

    # Save the updated DataFrame to the same CSV file
    updated_csv_path = os.path.join(csv_directory, 'unique_ingredients.csv')
    reshaped_df.to_csv(updated_csv_path, index=False, header=False)

def remove_searchword(df, searchword):
    df = df.apply(lambda x: x.str.replace(searchword, ''))
    # Save the updated DataFrame to the same CSV file
    updated_csv_path = os.path.join(csv_directory, 'unique_ingredients3.csv')
    df.to_csv(updated_csv_path, index=False, header=False)


csv_directory = "C:/Users/yewji/FYP_20297501/server/recipe_matching/bert"
print("1. Create new ingredients.csv")
print("2. Preprocess ingredients.csv")
print("3. Remove duplicates")
print("4. Update df")
print("5. Remove searchword")
selection = int(input("Select an option: "))
if selection == 1:
    create_ingredients_csv(csv_directory)

elif selection == 2:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/ingredients.csv")
    preprocess_dataset(df, csv_directory)

elif selection == 3:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/unique_ingredients.csv")
    remove_duplicates(df, csv_directory)

elif selection == 4:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/unique_ingredients2.csv")
    update_df(df, csv_directory)
elif selection == 5:
    df = pd.read_csv("C:/Users/yewji/FYP_20297501/server/recipe_matching/bert/unique_ingredients2.csv")
    searchword = input("Enter searchword to remove: ")
    remove_searchword(df, searchword)
else: 
    print("Invalid selection")
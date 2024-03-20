import pandas as pd
import re

def create_csv(df):

    # Create new csv with title, ingredients, and directions
    df = df[['title', 'ingredients', 'directions']]

    # Drop [] from ingredients, directions and titles
    df['ingredients'] = df['ingredients'].str.replace('[', '').str.replace(']', '')
    df['directions'] = df['directions'].str.replace('[', '').str.replace(']', '')
    df['title'] = df['title'].str.replace('[', '').str.replace(']', '')

    # Remove NaN values
    df = df.dropna()
    
    # Save new csv
    save_path = 'server/recipe_recommendation/t5/dataset_backup/recipes_dataset.csv'
    df.to_csv(save_path, index=False)
    return df

def get_phrases_to_remove():
    df = pd.read_csv('server/recipe_recommendation/t5/dataset_backup/sorted_ingredients.csv')

    phrases_to_remove = df['Tertiary'].tolist()
    # print(phrases_to_remove)
    # print(phrases_to_remove[0])
    # print(phrases_to_remove[1])
    # print(phrases_to_remove[2])

    return phrases_to_remove

def remove_phrases(ingredient):

    phrases_to_remove = get_phrases_to_remove()
    
    for phrase in phrases_to_remove:
        # Remove phrase from ingredient if it exists
        if phrase in ingredient:
            ingredient = ingredient.replace(phrase, '')
    return ingredient

csv_path = 'server/recipe_recommendation/t5/dataset_backup/cake.csv'
sorted_ingredients_csv = 'server/recipe_recommendation/t5/dataset_backup/sorted_ingredients.csv'
original_df = pd.read_csv(csv_path)

new_df = create_csv(original_df)
new_df['ingredients'] = new_df['ingredients'].apply(remove_phrases)
new_df.to_csv('server/recipe_recommendation/t5/dataset_backup/cleaned_recipes_dataset.csv', index=False)
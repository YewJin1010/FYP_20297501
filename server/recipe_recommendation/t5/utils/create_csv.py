import pandas as pd

def create_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Create new csv with title, ingredients, and directions
    df = df[['title', 'ingredients', 'directions']]

    # Drop [] from ingredients and directions
    df['ingredients'] = df['ingredients'].str.replace('[', '').str.replace(']', '')
    df['directions'] = df['directions'].str.replace('[', '').str.replace(']', '')

    # Wrap title in "" and add + after it
    df['title'] = '"' + df['title'] + '" + '  

    # Combine title and directions
    df['title_directions'] = df['title'] + df['directions']

    # Drop title and directions
    df = df.drop(['title', 'directions'], axis=1)

    # Save new csv
    save_path = 'server/recipe_recommendation/t5/dataset_backup/recipes_t5.csv'
    df.to_csv(save_path, index=False)

csv_path = 'server/recipe_recommendation/t5/dataset_backup/recipes.csv'
create_csv(csv_path)
import pandas as pd

def create_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Create new csv with title, ingredients, and directions
    df = df[['title', 'ingredients', 'directions']]

    # Drop [] from ingredients, directions and titles
    df['ingredients'] = df['ingredients'].str.replace('[', '').str.replace(']', '')
    df['directions'] = df['directions'].str.replace('[', '').str.replace(']', '')
    df['title'] = df['title'].str.replace('[', '').str.replace(']', '')
    
    # Save new csv
    save_path = 'server/recipe_recommendation/t5/dataset/recipes_dataset.csv'
    df.to_csv(save_path, index=False)

csv_path = 'server/recipe_recommendation/t5/dataset_backup/recipes.csv'
create_csv(csv_path)
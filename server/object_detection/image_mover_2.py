import pandas as pd
import shutil
import os

def organize_images_by_ingredients(csv_path, source_dir, destination_base_dir, new_csv_path, ingredients_to_search):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Loop through specified ingredients
    for ingredient in ingredients_to_search:
        # Check if the ingredient is in the CSV
        if ingredient in df.columns:
            # Filter rows where the specified ingredient has a value of 1
            images_with_ingredient = df[df[ingredient] == 1]['filename'].tolist()

            # Create a folder for the ingredient if it doesn't exist
            destination_dir = os.path.join(destination_base_dir, ingredient)
            os.makedirs(destination_dir, exist_ok=True)

            # Organize images into the destination folder
            for image in images_with_ingredient:
                source_path = os.path.join(source_dir, image)
                destination_path = os.path.join(destination_dir, image)

                # Check if the image already exists in the destination folder
                if not os.path.exists(destination_path):
                    shutil.copyfile(source_path, destination_path)

                else:
                    print(f"Image '{image}' already exists in '{destination_dir}'")

        else:
            print(f"{ingredient} not found in the CSV file.")

    print(f"Updated CSV saved to {new_csv_path}")

# Example usage
csv_file_path = 'C:/Users/miku/Documents/Yew Jin/datasets/FOOD-INGREDIENTS/test/_classes.csv'
source_directory = 'C:/Users/miku/Documents/Yew Jin/datasets/FOOD-INGREDIENTS/test'
destination_base_directory = 'C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/test'
new_csv_file_path = 'C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/test/_classes.csv'

ingredients_to_search = [' Apple', ' Avocado', ' Banana', ' Carrot', ' Cassava']

organize_images_by_ingredients(csv_file_path, source_directory, destination_base_directory, new_csv_file_path, ingredients_to_search)

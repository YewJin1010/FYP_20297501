import pandas as pd
import shutil
import os

def list_classes(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get the column names (classes)
    classes = df.columns.tolist()

    return classes

def organize_images_by_ingredients(csv_path, source_dir, destination_dir, ingredients_to_search):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Loop through specified ingredients
    for ingredient in ingredients_to_search:
        # Check if the ingredient is in the CSV
        if ingredient in df.columns:
            # Set the base destination directory
            base_destination_dir = destination_dir

            # Filter rows where the specified ingredient has a value of 1
            images_with_ingredient = df[df[ingredient] == 1]['filename'].tolist()

            cleaned_ingredient = ingredient.strip().lower()

            # Create a folder for the ingredient if it doesn't exist
            current_destination_dir = os.path.join(base_destination_dir, cleaned_ingredient)
            os.makedirs(current_destination_dir, exist_ok=True)

            # Organize images into the destination folder
            for image in images_with_ingredient:
                source_path = os.path.join(source_dir, image)
                destination_path = os.path.join(current_destination_dir, image)

                # Check if the image exists in the source directory
                if os.path.exists(source_path):
                    # Check if the image already exists in the destination folder
                    if not os.path.exists(destination_path):
                        shutil.copyfile(source_path, destination_path)
                    else:
                        print(f"Image '{image}' already exists in '{current_destination_dir}'")
                else:
                    print(f"Image '{image}' does not exist in '{source_dir}'")

        else:
            print(f"{ingredient} not found in the CSV file.")

def run_for_all_directories():
    source_base_directory = 'C:/Users/yewji/fyp3/datasets/Keras_Datasets/food_ingredients_multiclass/'
    destination_base_directory = 'C:/Users/yewji/FYP_20297501/server/object_detection/'

    directories = ['test', 'train', 'valid']

    for directory in directories:
        csv_file_path = os.path.join(source_base_directory, directory, '_classes.csv')
        ingredients_to_search = [' Apple', ' Avocado', ' Banana', ' Carrot']

        source_directory = os.path.join(source_base_directory, directory)
        destination_directory = os.path.join(destination_base_directory, directory)

        print(f"\nProcessing directory: {directory}")
        print("Source Directory:", source_directory)
        print("Destination Directory:", destination_directory)
        
        organize_images_by_ingredients(csv_file_path, source_directory, destination_directory, ingredients_to_search)

# Example usage
csv_file_path = 'C:/Users/yewji/fyp3/datasets/Keras_Datasets/food_ingredients_multiclass/test/_classes.csv'

print("1. List classes")
print("2. Organize images by ingredients")
selection = int(input())
if selection == 1:
    print(list_classes(csv_file_path))
elif selection == 2:
    run_for_all_directories()

import pandas as pd
import shutil
import os
import re

def list_classes(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Get the column names (classes)
    classes = df.columns.tolist()

    return classes


def sanitize_filename(filename):
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def count_images_in_subfolders(base_directory, dataset_type):
    dataset_path = os.path.join(base_directory, dataset_type)
    
    # Ensure the dataset type is valid
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_type}' directory not found.")
        return

    print(f"\nCounting images in '{dataset_type}' directory:")
    
    # Loop through subdirectories
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Count the number of files in the subdirectory
            file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            
            print(f"{subfolder}: {file_count} images")

def organise_images_by_ingredients(csv_path, source_dir, destination_dir, ingredients_to_search):
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

            # Remove white spaces and convert to lower case
            cleaned_ingredient = ingredient.strip().lower().replace(" ", "")


            # Create a folder for the ingredient if it doesn't exist
            current_destination_dir = os.path.join(base_destination_dir, cleaned_ingredient)
            os.makedirs(current_destination_dir, exist_ok=True)

            # Organize images into the destination folder
            for image in images_with_ingredient:
                source_path = os.path.join(source_dir, image)
                
                # Sanitize the filename
                sanitized_filename = sanitize_filename(image)
                destination_path = os.path.join(current_destination_dir, sanitized_filename)

                # Check if the image exists in the source directory
                if os.path.exists(source_path):
                    # Ensure the destination folder exists
                    os.makedirs(current_destination_dir, exist_ok=True)

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
    source_base_directory = 'C:/Users/miku/Documents/Yew Jin/datasets/FoodDataSet1/'
    destination_base_directory = 'C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/'

    directories = ['test', 'train', 'valid']

    for directory in directories:
        csv_file_path = os.path.join(source_base_directory, directory, '_classes.csv')
        ingredients_to_search = [ ' ketchup', ' kiwi', ' lemon',' mandarine', ' mango', ' mayonnaise',' mozzarella',' muffin',' mustard',' nectarine',' nuts',  ' onion', ' orange', ' parmesan',  ' parsley',  ' plums', ' pomegranate', ' potato', ' praline_n_s', ' pumpkin',' quinoa', ' raisins_dried', ' raspberries', ' rice',' salad_dressing',' sesame_seeds',' sweet_potato',' tart_n_s',' tea',' tofu', ' tomato', ' tomato_sauce',' walnut', ' water', ' water_with_lemon_juice', ' watermelon_fresh', ' white_coffee', ' wine_red',' yogurt'
        ]
        source_directory = os.path.join(source_base_directory, directory)
        destination_directory = os.path.join(destination_base_directory, directory)

        print(f"\nProcessing directory: {directory}")
        print("Source Directory:", source_directory)
        print("Destination Directory:", destination_directory)
        
        organise_images_by_ingredients(csv_file_path, source_directory, destination_directory, ingredients_to_search)

# Example usage
csv_file_path = 'C:/Users/miku/Documents/Yew Jin/datasets/FoodDataSet1/test/_classes.csv'

print("1. List classes")
print("2. Organize images by ingredients")
print("3. Count images in subfolders")
selection = int(input())
if selection == 1:
    print(list_classes(csv_file_path))
elif selection == 2:
    run_for_all_directories()
elif selection == 3:
    base_directory = 'C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/'
    dataset_types = ['test', 'train', 'valid']
    for dataset_type in dataset_types:
        count_images_in_subfolders(base_directory, dataset_type)

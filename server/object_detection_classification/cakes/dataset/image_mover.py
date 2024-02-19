import pandas as pd
import shutil
import os

def print_images_with_ingredient(csv_path, ingredient, source_dir, destination_dir, new_csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Check if the ingredient is in the first row
    if ingredient in df.columns:
        
        # Filter rows where the specified ingredient has a value of 1
        images_with_ingredient = df[df[ingredient] == 1]['filename'].tolist()

        # Print the images with the specified ingredient
        if images_with_ingredient:
            for image in images_with_ingredient:
                print(image)
                source_path = os.path.join(source_dir, image)
                destination_path = os.path.join(destination_dir, image)
                shutil.copyfile(source_path, destination_path)

                # Append the image name to the 'filename' column in the new CSV file
                with open(new_csv_path, 'a') as f:
                    f.write(image + '\n')
            print(f"Images and filenames copied to {destination_dir} and updated CSV saved to {new_csv_path}")
        else:
            print(f"No images found with {ingredient}")
    else:
        print(f"{ingredient} not found in the CSV file.")
# Example usage
ingredients_to_search = [ 
    ' butter',' egg',' milk'
]
csv_file_path = 'C:/Users/yewji/fyp/datasets/Ingredients.v1/test/_classes.csv'
source_directory = 'C:/Users/yewji/fyp/datasets/Ingredients.v1/test'
destination_directory = 'C:/Users/yewji/fyp/datasets/cakes/dataset/test'
new_csv_file_path = 'C:/Users/yewji/fyp/datasets/cakes/dataset/test/_classes.csv'

for ingredient_to_search in ingredients_to_search:
    print_images_with_ingredient(csv_file_path, ingredient_to_search, source_directory, destination_directory, new_csv_file_path)
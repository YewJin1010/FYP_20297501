import pandas as pd
import shutil
import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
    subfolder_counts = {}
    
    dataset_path = os.path.join(base_directory, dataset_type)
    
    # Ensure the dataset type is valid
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_type}' directory not found.")
        return

    print(f"\nCounting images in '{dataset_type}' directory:")
    
    total_file_count = 0
    # Loop through subdirectories
    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Count the number of files in the subdirectory
            file_count = len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])
            subfolder_counts[subfolder] = file_count  # Update the dictionary with the count
            total_file_count += file_count
            print(f"{subfolder}: {file_count} images")
    
    print(f"Total {dataset_type} images: {total_file_count}")

    return subfolder_counts


def process_images(directory):
    # Get a list of image files in the directory
    image_files = [f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png'))]

    for image_file in image_files:
        # Open the image
        image_path = os.path.join(directory, image_file)
        original_image = Image.open(image_path)

        # Convert to RGB if the image is in RGBA mode
        if original_image.mode == 'RGBA':
            original_image = original_image.convert('RGB')

        # Perform rotations (90, 180, and 270 degrees)
        for angle in [90, 180, 270]:
            rotated_image = original_image.rotate(angle)
            rotated_filename = f"{os.path.splitext(image_file)[0]}_rotated_{angle}.jpg"
            rotated_path = os.path.join(directory, rotated_filename)
            rotated_image.save(rotated_path)

        # Perform horizontal and vertical flips
        flipped_horizontal = original_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_horizontal_filename = f"{os.path.splitext(image_file)[0]}_flipped_horizontal.jpg"
        flipped_horizontal_path = os.path.join(directory, flipped_horizontal_filename)
        flipped_horizontal.save(flipped_horizontal_path)

        flipped_vertical = original_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_vertical_filename = f"{os.path.splitext(image_file)[0]}_flipped_vertical.jpg"
        flipped_vertical_path = os.path.join(directory, flipped_vertical_filename)
        flipped_vertical.save(flipped_vertical_path)
    print("done")

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
    destination_base_directory = 'C:/Users/yewji/server/object_detection/'

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

def rename_images(directory, dataset_type):
    dataset_path = os.path.join(directory, dataset_type)

    # Ensure the dataset type is valid
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_type}' directory not found.")
        return

    print(f"\nRenaming all images in '{dataset_type}' directory:")

    # List all subdirectories in the dataset path
    subdirectories = [subdir for subdir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subdir))]

    # Loop through subdirectories
    for subfolder in subdirectories:
        subfolder_path = os.path.join(dataset_path, subfolder)

        # List all files in the subdirectory
        image_files = os.listdir(subfolder_path)

        # Separate files into correctly and incorrectly named arrays
        correct_images = [image for image in image_files if image.lower().endswith(('.png', '.jpg', '.jpeg')) and "_" in image]
        incorrect_images = [image for image in image_files if 'Image_' in image]
        
        # Sort correctly named images
        correct_images.sort()

        # Initialize a counter for images
        image_counter = 0

        # Rename correctly named images
        for image in correct_images:
            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_counter += 1
                new_name = f"{subfolder}_{image_counter}.jpg"  # Assuming jpg extension, change if needed

                new_path = os.path.join(subfolder_path, new_name)

                # Check if the destination file already exists
                if not os.path.exists(new_path):
                    old_path = os.path.join(subfolder_path, image)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {image} to {new_name}")
                else:
                    print(f"Skipping: {new_name} already exists.")

        # Use the last image_counter value for incorrectly named images
        for image in incorrect_images:
            if image.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_counter += 1
                new_name = f"{subfolder}_{image_counter}.jpg"  # Assuming jpg extension, change if needed

                new_path = os.path.join(subfolder_path, new_name)

                # Check if the destination file already exists
                if not os.path.exists(new_path):
                    old_path = os.path.join(subfolder_path, image)
                    os.rename(old_path, new_path)
                    print(f"Renamed: {image} to {new_name}")
                else:
                    print(f"Skipping: {new_name} already exists.")

    print("Renaming done.")
    
def create_classes_csv(directory, dataset_type):
    dataset_path = os.path.join(directory, dataset_type)

    # Ensure the dataset type is valid
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_type}' directory not found.")
        return

    print(f"\nCreating _classes.csv file in '{dataset_type}' directory:")

    # List all subdirectories in the dataset path
    subdirectories = [subdir for subdir in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, subdir))]

    # Create an empty DataFrame with the 'filename' and class columns
    df = pd.DataFrame(columns=['filename'] + subdirectories)

    # Loop through subdirectories
    for subfolder in subdirectories:
        subfolder_path = os.path.join(dataset_path, subfolder)

        # List image files in the subdirectory
        #image_files = [os.path.join(subfolder, f) for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Update the 'filename' column with image file names
        df = pd.concat([df, pd.DataFrame({'filename': image_files})], ignore_index=True)

        # Set the specific class column to 1 for each image
        df.loc[df['filename'].isin(image_files), subfolder] = 1

    # Fill NaN values with 0
    df = df.fillna(0)

    # Write the DataFrame to the _classes.csv file
    csv_path = os.path.join(dataset_path, '_classes.csv')
    df.to_csv(csv_path, index=False)

    print(f"Created _classes.csv file: {csv_path}")

csv_file_path = 'C:/Users/miku/Documents/Yew Jin/datasets/FoodDataSet1/test/_classes.csv'

print("1. List classes")
print("2. Organize images by ingredients")
print("3. Count images in subfolders")
print("4. Rotate / Clone Images")
print("5. Rename Images")
print("6. Create _classes.csv file")
selection = int(input())
if selection == 1:
    print(list_classes(csv_file_path))
elif selection == 2:
    run_for_all_directories()
elif selection == 3:
    base_directory = 'C:/Users/yewji/FYP_20297501/server/object_detection/dataset'
    dataset_types = ['test', 'train', 'valid']

    counts_by_dataset = {}
    for dataset_type in dataset_types:
        counts_by_dataset[dataset_type] = count_images_in_subfolders(base_directory, dataset_type)

    # Plotting the bar chart
    bar_width = 0.25
    index = np.arange(len(counts_by_dataset['test']))

    colors = ['red', 'green', 'blue']
    for i, dataset_type in enumerate(dataset_types):
        counts = [counts_by_dataset[dataset_type][subfolder] for subfolder in counts_by_dataset['test']]
        plt.bar(index + i * bar_width, counts, bar_width, label=dataset_type, color=colors[i])

    plt.xlabel('Subfolders')
    plt.ylabel('Number of Images')
    plt.xticks(index + bar_width / 2, counts_by_dataset['test'].keys())  # Adjusted the position to center the bars
    plt.legend()

    # Adding spacing between bottom text
    plt.subplots_adjust(bottom=0.2)

    # Adjusting heights of text labels
    for i, tick in enumerate(plt.gca().xaxis.get_major_ticks()):
        if i % 3 == 0:  # Original position
            tick.set_pad(0)
        elif i % 3 == 1:  # Lowered position
            tick.set_pad(15)
        else:  # Lowest position
            tick.set_pad(30)

    plt.show()

elif selection == 4:
    directory = 'C:/Users/yewji/server/object_detection/web_scrapping/turnip_istockphoto'
    process_images(directory)

elif selection == 5:
    directory = 'C:/Users/yewji/FYP_20297501/server/object_detection/'
    dataset_types = ['test', 'train', 'valid']

    for dataset_type in dataset_types:
        rename_images(directory, dataset_type)

elif selection == 6:
    directory = 'C:/Users/yewji/FYP_20297501/server/object_detection/'
    dataset_types = ['test', 'train', 'valid']

    for dataset_type in dataset_types:
        create_classes_csv(directory, dataset_type)

else :
    print("Invalid selection")
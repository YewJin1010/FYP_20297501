import yaml
import os
import pandas as pd

def read_data_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def get_classes_from_data_yaml(file_path):
    data = read_data_yaml(file_path)
    if 'names' in data:
        return data['names']
    else:
        return None

def get_classes(dataset_path):
    dataset_path = 'server/object_detection_classification/yolo_dataset'
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    unique_classes = get_classes_from_data_yaml(data_yaml_path)
    print("Unique classes in the dataset:", unique_classes)

    # Read the ingredients CSV into a DataFrame
    ingredients_df = pd.read_csv('server/database/cleaned_ingredients.csv')

    # Concatenate all values from the three columns into a single string
    all_ingredients = ingredients_df['Primary'].fillna('') + ' ' + \
                    ingredients_df['Secondary'].fillna('')

    # Initialize an empty list to store matched classes
    matched_classes = []

    # Check if any word from unique_classes exists in the concatenated string
    for word in unique_classes:
        if any(word.lower() in ingredient.lower() for ingredient in all_ingredients):
            matched_classes.append(word)

    print("Classes matched with ingredients:", matched_classes)

    # Remove classes
    classes_to_remove = input("Enter the classes to remove (comma-separated): ").split(',')
    cleaned_classes = [cls for cls in matched_classes if cls not in classes_to_remove]

    print("Unique classes after removal:", cleaned_classes)
    return cleaned_classes


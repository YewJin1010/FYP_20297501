import os
import pandas as pd

def get_unique_classes_in_dataset(dataset_path):
    unique_classes = []
    for directory in ['train', 'test', 'valid']:
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)
        unique_classes.extend(df['class'].unique().tolist())
    unique_classes = list(set(unique_classes))  # Remove duplicates
    return unique_classes


def get_classes(dataset_path):
    #dataset_path = 'server/object_detection_classification/multiclass_dataset'
    unique_classes = get_unique_classes_in_dataset(dataset_path)
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
    print("Enter none if you don't want to remove any classes.")
    cleaned_classes = [cls for cls in matched_classes if cls not in classes_to_remove]

    print("Unique classes after removal:", cleaned_classes)
    return cleaned_classes
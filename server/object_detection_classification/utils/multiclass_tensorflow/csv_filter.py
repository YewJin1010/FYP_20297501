import pandas as pd
import os 

def count_classes(dataset_path):
    for directory in ['train', 'test', 'valid']:
            # Read annotations CSV file
            csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
            df = pd.read_csv(csv_file)

            print(f"Classes in {directory} dataset")

            # Convert class column to lowercase
            df['class'] = df['class'].str.lower()

            # show classes in the dataset
            print(df['class'].unique())

            # Number of classes in the dataset
            print(len(df['class'].unique()), "unique classes\n")

            # Number of times each class appears in the dataset
            class_counts = df['class'].value_counts()
            print("Number of times each class appears in the dataset:")
            print(class_counts)

            # Write class counts into a csv
            class_counts.to_csv(f'server/object_detection_classification/utils/multiclass_tensorflow/{directory}_class_counts.csv')

def get_classes_present_in_all_datasets(dataset_path):
    # Initialize sets to store unique classes in each dataset
    unique_classes_train = set()
    unique_classes_test = set()
    unique_classes_valid = set()

    # Iterate through the directories
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Convert class column to lowercase
        df['class'] = df['class'].str.lower()

        # Add unique classes to respective sets
        if directory == 'train':
            unique_classes_train.update(df['class'].unique())
        elif directory == 'test':
            unique_classes_test.update(df['class'].unique())
        elif directory == 'valid':
            unique_classes_valid.update(df['class'].unique())

    # Find the intersection of classes present in all three datasets
    classes_present_in_all = unique_classes_train.intersection(unique_classes_test, unique_classes_valid)
    
    return list(classes_present_in_all)

def count_class_occurrences(dataset_path, classes_present_in_all):
    class_counts = {}

    # Iterate through the directories
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Convert class column to lowercase
        df['class'] = df['class'].str.lower()

        # Count occurrences of classes present in all datasets
        for class_name in classes_present_in_all:
            class_counts.setdefault(class_name, {})
            class_counts[class_name][directory] = df[df['class'] == class_name].shape[0]

    return class_counts

def get_classes_with_shared_images(dataset_path):
    # Initialize an empty dictionary to store classes with shared images and their counts
    classes_with_shared_images = {}

    # Iterate through the directories
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Group DataFrame by 'filename' and count unique classes for each image
        image_class_counts = df.groupby('filename')['class'].nunique()

        # Filter groups with more than 1 unique class and get corresponding classes
        multiple_class_images = image_class_counts[image_class_counts > 1]
        for image_filename in multiple_class_images.index:
            shared_classes = set(df[df['filename'] == image_filename]['class'])
            for cls in shared_classes:
                if cls not in classes_with_shared_images:
                    classes_with_shared_images[cls] = {'train': 0, 'test': 0, 'valid': 0}
                classes_with_shared_images[cls][directory] += 1

    return classes_with_shared_images

def remove_classes_from_all_datasets(dataset_path, classes_to_remove):
    # Iterate through the directories
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Convert class column to lowercase
        df['class'] = df['class'].str.lower()

        # Remove rows with specified classes
        df = df[~df['class'].isin(classes_to_remove)]

        # Write back to CSV file
        df.to_csv(csv_file, index=False)

dataset_path = 'server/object_detection_classification/multiclass_dataset'
classes_present_in_all_datasets = get_classes_present_in_all_datasets(dataset_path)
print("Classes present in all three datasets:", classes_present_in_all_datasets)
print("Number of classes present in all three datasets:", len(classes_present_in_all_datasets))

class_occurrences = count_class_occurrences(dataset_path, classes_present_in_all_datasets)
print("\nNumber of occurrences of classes present in all datasets:")
for class_name, counts in class_occurrences.items():
    print(f"{class_name}: {counts}")

# Get the classes that share images
shared_classes = get_classes_with_shared_images(dataset_path)
print("\nClasses that share images and their counts:")
for cls, counts in shared_classes.items():
    print(f"{cls}: Train - {counts['train']}, Test - {counts['test']}, Valid - {counts['valid']}")


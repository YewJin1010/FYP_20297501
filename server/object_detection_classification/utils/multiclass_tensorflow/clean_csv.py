import pandas as pd
import os 

# read csv
dataset_path = 'server/object_detection_classification/multiclass_dataset'

def clean_by_images(dataset_path):
    for directory in ['train', 'test', 'valid']:
        print(f"Cleaning {directory} dataset...")
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Create a list to store indices of rows to remove
        rows_to_remove = []

        # Iterate through each row in the DataFrame
        for index, row in df.iterrows():
            image_path = os.path.join(dataset_path, directory, row['filename'])

            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Image {row['filename']} does not exist in {directory}. Removing row...")
                rows_to_remove.append(index)

        # Remove rows from DataFrame
        df_cleaned = df.drop(rows_to_remove)

        # Save cleaned DataFrame back to CSV file
        df_cleaned.to_csv(csv_file, index=False)

        print(f"Finished cleaning {directory} dataset.")
        print(f"New length of df: {len(df_cleaned)}")

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

def clean_by_df(dataset_path): 
    for directory in ['train', 'test', 'valid']:
        print(f"Cleaning {directory} dataset...")
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Get list of existing image filenames
        image_files = os.listdir(os.path.join(dataset_path, directory))

        # Check if filename exists in DataFrame, if not, remove image
        for filename in image_files:
            if filename not in df['filename'].values:
                image_path = os.path.join(dataset_path, directory, filename)
                os.remove(image_path)
                print(f"Removed image: {image_path}")

        # Remove rows from DataFrame if corresponding image does not exist
        df = df[df['filename'].isin(image_files)]
        df.to_csv(csv_file, index=False)
        print(f"Length of df after removal: {len(df)}")


print("1. Clean by images")
print("2. Clean by DataFrame")
print("3. Remove a classes from all datasets")
choice = input("Enter your choice (1, 2, or 3): ")

dataset_path = 'server/object_detection_classification/multiclass_dataset'
if choice == '1':
    clean_by_images(dataset_path)
elif choice == '2':
    clean_by_df(dataset_path)
elif choice == '3':
    remove_classes = input("Enter the classes to remove (comma-separated): ").split(',')
    remove_classes_from_all_datasets(dataset_path, remove_classes)
else:
    print("Invalid choice.")

#  olive oil,noodle,pea,ice,pear,ginger,d,wheat
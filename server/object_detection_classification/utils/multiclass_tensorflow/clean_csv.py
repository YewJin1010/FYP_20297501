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
        original_length = len(df)

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
        print("Original length of df: ", original_length)
        print(f"New length of df: {len(df_cleaned)}")


clean_by_images(dataset_path)
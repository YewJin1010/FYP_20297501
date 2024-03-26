import os
import pandas as pd
import shutil

from get_multiclasses import get_classes

dataset = 'ingredients v1i'
dataset_path = f'C:/Users/miku/Documents/Yew Jin/xml_dataset/{dataset}' 
desination_path = 'server/object_detection_classification/multiclass_dataset'

# craete new dataset
def create_new_dataset(classes_to_transfer, dataset_path, destination_path):
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)

        # Filter DataFrame to include only specified classes
        df_filtered = df[df['class'].isin(classes_to_transfer)]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(destination_path, directory), exist_ok=True)
        
        # Save filtered DataFrame to new CSV file
        output_csv_file = os.path.join(destination_path, directory, '_annotations.csv')
        if os.path.exists(output_csv_file):
            # Append to existing CSV file if it exists
            df_filtered.to_csv(output_csv_file, mode='a', header=False, index=False)
        else:
            # Otherwise, create a new CSV file
            df_filtered.to_csv(output_csv_file, index=False)

        # Copy images for each class
        for cls in classes_to_transfer:
            for index, row in df[df['class'] == cls].iterrows():
                image_path = os.path.join(dataset_path, directory, row['filename'])
                destination_image_path = os.path.join(destination_path, directory, row['filename'])
                
                # Check if destination image file exists
                if os.path.exists(destination_image_path):
                    print(f"Destination image {destination_image_path} already exists. Skipping...")
                    continue
                
                try:
                    # Copy image using shutil.copy()
                    shutil.copy(image_path, destination_image_path)
                    print(f"Image {image_path} copied to {destination_image_path}")
                except FileNotFoundError:
                    print(f"Image {image_path} does not exist. Skipping...")

classes_to_transfer = get_classes(dataset_path)
print("Classes to transfer:", classes_to_transfer)
create_new_dataset(classes_to_transfer, dataset_path, desination_path)
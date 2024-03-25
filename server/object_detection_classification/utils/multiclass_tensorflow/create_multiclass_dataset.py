import os
import pandas as pd
import shutil

from unique_classes import get_classes

dataset_path = 'server/object_detection_classification/multiclass_dataset'
desination_path = 'server/object_detection_classification/transfer_dataset'

classes_to_transfer = get_classes(dataset_path)
print("Classes to transfer:", classes_to_transfer)

# craete new dataset
def create_new_dataset(classes_to_transfer, dataset_path, destination_path):
    for directory in ['train', 'test', 'valid']:
        # Read annotations CSV file
        csv_file = os.path.join(dataset_path, directory, '_annotations.csv')
        df = pd.read_csv(csv_file)
        
        # Filter DataFrame to include only specified classes
        df = df[df['class'].isin(classes_to_transfer)]
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.join(destination_path, directory), exist_ok=True)
        
        # Save filtered DataFrame to new CSV file
        df.to_csv(os.path.join(destination_path, directory, '_annotations.csv'), index=False)

        # Copy images for each class
        for cls in classes_to_transfer:
            for index, row in df[df['class'] == cls].iterrows():
                image_path = os.path.join(dataset_path, directory, row['filename'])
                destination_image_path = os.path.join(destination_path, directory, row['filename'])
                
                # Check if image file exists
                if os.path.exists(image_path):
                    # Copy image using shutil.copy()
                    shutil.copy(image_path, destination_image_path)
                else:
                    print(f"Image {image_path} does not exist. Skipping...")



create_new_dataset(classes_to_transfer, dataset_path, desination_path)
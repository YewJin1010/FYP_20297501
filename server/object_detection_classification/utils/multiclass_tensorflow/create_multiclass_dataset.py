import os
import pandas as pd
import shutil

from get_multiclasses import get_classes

dataset_path = 'C:/Users/miku/Documents/Yew Jin/xml_dataset/IngredientDetector v4i' 
desination_path = 'server/object_detection_classification/multiclass_dataset'

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
                
                # Check if destination image file exists
                if os.path.exists(destination_image_path):
                    print(f"Destination image {destination_image_path} already exists. Skipping...")
                    continue
                
                # Check if image file exists
                if os.path.exists(image_path):
                    # Check if the current row already exists in the destination DataFrame
                    existing_rows = df[(df['filename'] == row['filename']) & (df['class'] == row['class'])]
                    if len(existing_rows) > 1:
                        # Skip copying the image if all data in the row are the same as an existing row
                        if all(existing_rows.iloc[0] == existing_rows.iloc[i] for i in range(1, len(existing_rows))):
                            print(f"Skipping image {image_path} as all data in the row are the same as an existing row.")
                            continue
                        
                    # Copy image using shutil.copy()
                    shutil.copy(image_path, destination_image_path)
                else:
                    print(f"Image {image_path} does not exist. Skipping...")

classes_to_transfer = get_classes(dataset_path)
print("Classes to transfer:", classes_to_transfer)
create_new_dataset(classes_to_transfer, dataset_path, desination_path)
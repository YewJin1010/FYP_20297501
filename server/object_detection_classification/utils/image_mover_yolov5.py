import os
import yaml

# Read the data.yaml file
def read_yaml(data_yaml_path):
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def remove_labels(data, labels_path, classes_to_remove):
    for filename in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, filename)
        
        # Read the first number from the label txt file
        with open(label_file_path, 'r') as label_file:
            # Read the lines and split them into elements
            lines = label_file.readlines()
            
            # Filter out lines with at least one element
            lines = [line.split() for line in lines if line.strip()]
            
            if lines:
                # Process each line
                for i, line_elements in enumerate(lines):
                    if len(line_elements) > 0:
                        first_number = int(line_elements[0])
                        
                        # Check if the class should be removed
                        if 0 <= first_number < len(data['names']):
                            class_name = data['names'][first_number].lower()
                            if class_name in [label.lower() for label in classes_to_remove]:
                                
                                print(f"Removing: {class_name} from {label_file_path}")
                                # Remove the line from the list
                                lines[i] = None
                        else:
                            print(f"Invalid index: {first_number} in {label_file_path}")
                        
                # Filter out None values (lines to be removed)
                lines = [line for line in lines if line is not None]
                
                # Write the modified lines back to the label file
                with open(label_file_path, 'w') as updated_label_file:
                    updated_label_file.writelines('\n'.join([' '.join(line) for line in lines]))
                    
                print(f"Removed lines in: {label_file_path}")
            else:
                print(f"Skipping: {label_file_path}, no content or empty lines")
    
    print(f"Finished removing labels in: {labels_path}")


def remove_empty_labels(labels_path):
    for filename in os.listdir(labels_path):
        label_file_path = os.path.join(labels_path, filename)

        with open(label_file_path, 'r') as label_file:
            lines = label_file.readlines()
            lines = [line for line in lines if line.strip()]
            
        if not lines:
                os.remove(label_file_path)
                print(f"Removed: {label_file_path}")
    print(f"Finished removing empty labels in: {labels_path}")

def remove_images(images_path, labels_path):
    for filename in os.listdir(images_path):
        image_file_path = os.path.join(images_path, filename)
        label_file_path = os.path.join(labels_path, filename.replace('.jpg', '.txt'))

        if not os.path.exists(label_file_path):
            try:
                os.remove(image_file_path)
                print(f"Removed image without label: {image_file_path}")
            except Exception as e:
                print(f"Error removing image without label: {image_file_path}, {e}")

    print(f"Finished removing images without labels in: {images_path}")


def remove_images_and_labels(directory, dataset_type, classes_to_remove):
    data = read_yaml(os.path.join(directory, 'data.yaml'))
    labels_path = os.path.join(directory, dataset_type, 'labels')
    images_path = os.path.join(directory, dataset_type, 'images')
    remove_labels(data, labels_path, classes_to_remove)
    remove_empty_labels(labels_path)
    remove_images(images_path, labels_path)

def rename_labels(directory, dataset_types, new_labels):
    data = read_yaml(os.path.join(directory, 'data.yaml'))

    # Create a mapping from old class number to class name
    old_class_mapping = {str(i): data['names'][i] for i in range(len(data['names']))}

    for dataset_type in dataset_types:
        labels_path = os.path.join(directory, dataset_type, 'labels')
        
        for filename in os.listdir(labels_path):
            label_file_path = os.path.join(labels_path, filename)

            # Read the contents of the label file
            with open(label_file_path, 'r') as label_file:
                lines = label_file.readlines()

            # Create a new label file with renumbered classes
            with open(label_file_path, 'w') as label_file:
                for line in lines:
                    # Split the line into elements
                    line_elements = line.split()
                    
                    # Check if the line is not empty and has at least one element
                    if line_elements:
                        old_class = line_elements[0]
                        
                        # Map old class number to class name
                        class_name = old_class_mapping.get(old_class, 'unknown')
                        
                        # Map class name to new class number
                        try:
                            new_class = new_labels.index(class_name)
                            print(f"Renaming: {class_name} to {new_class}")
                        except ValueError:
                            print(f"Class {class_name} not found in new_labels. Skipping...")
                            continue
                        
                        # Update the class number in the line
                        line_elements[0] = str(new_class)
                        
                        new_line = ' '.join(line_elements)
                        label_file.write(new_line + '\n')
                    else:
                        label_file.write('\n')

    print(f"Finished renaming labels in: {labels_path}")

            

directory = 'C:/Users/yewji/FYP_20297501/server/object_detection/dataset/Detection of ingr v3i'
dataset_types = ['test', 'train', 'valid']
new_labels =['apple','Avocado', 'Brocolli', 'Honey', 'Lemon', 'egg', 'oil', 'orange']

print("1. Remove images and labels")
print("2. Rename labels")
choice = input("Enter choice: ")
if choice == '1':
    classes_to_remove = input("Enter the labels to remove: ").split()  # split by space
    for dataset_type in dataset_types:
        remove_images_and_labels(directory, dataset_type, classes_to_remove)

elif choice == '2':
    for dataset_type in dataset_types:
        rename_labels(directory, dataset_types, new_labels)
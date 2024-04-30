import shutil
import os

DIR_PATHS = {
    "Test Data": "server/object_detection_classification/dataset/test",
    "Train Data": "server/object_detection_classification/dataset/train",
    "Validation Data": "server/object_detection_classification/dataset/valid",
    "Raw Data": "server/object_detection_classification/dataset/raw_data"
}

def get_latest_number(raw_class_dir):
    """Get the latest number used for renaming within a class directory."""
    latest_number = 0
    for file in os.listdir(raw_class_dir):
        if file.endswith((".jpg", ".jpeg", ".png")):
            file_number = int(''.join(filter(str.isdigit, file.split('.')[0])))
            latest_number = max(latest_number, file_number)
    return latest_number

def copy_and_rename_images(dir_paths):
    # Create the 'raw_data' directory if it doesn't exist
    raw_data_dir = dir_paths["Raw Data"]
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    # Copy and rename images from 'test' and 'train' directories to 'raw_data'
    for dir_name, dir_path in dir_paths.items():
        if dir_name != "Raw Data" and os.path.exists(dir_path):
            for class_dir in os.listdir(dir_path):
                class_path = os.path.join(dir_path, class_dir)
                if os.path.isdir(class_path):
                    raw_class_dir = os.path.join(raw_data_dir, class_dir)
                    if not os.path.exists(raw_class_dir):
                        os.makedirs(raw_class_dir)
                    latest_number = get_latest_number(raw_class_dir)
                    for file in os.listdir(class_path):
                        if file.endswith((".jpg", ".jpeg", ".png")):
                            latest_number += 1
                            src_file = os.path.join(class_path, file)
                            dst_file = os.path.join(raw_class_dir, f"{class_dir}_{latest_number}.jpg")
                            shutil.copy(src_file, dst_file)

# Copy and rename images into 'raw_data' directory
copy_and_rename_images(DIR_PATHS)

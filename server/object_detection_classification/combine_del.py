import shutil
import os

# Define directory paths
DIR_PATHS = {
    "Test Data": "server/object_detection_classification/dataset/test",
    "Train Data": "server/object_detection_classification/dataset/train",
    "Validation Data": "server/object_detection_classification/dataset/valid",
    "Raw Data": "server/object_detection_classification/dataset/raw_data"
}

def combine_images(dir_paths):
    # Create the 'raw_data' directory if it doesn't exist
    raw_data_dir = dir_paths["Raw Data"]
    if not os.path.exists(raw_data_dir):
        os.makedirs(raw_data_dir)

    # Combine images from 'test', 'train', and 'valid' directories into 'raw_data'
    for dir_name, dir_path in dir_paths.items():
        if dir_name != "Raw Data":
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    src_file = os.path.join(root, file)
                    dst_file = os.path.join(raw_data_dir, file)
                    shutil.copy(src_file, dst_file)

# Combine images into 'raw_data' directory
combine_images(DIR_PATHS)

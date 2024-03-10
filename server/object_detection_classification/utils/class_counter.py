import os
import pandas as pd
import matplotlib.pyplot as plt

# directory paths
DIR_PATHS = {
    "Raw Data": "server/object_detection_classification/dataset/raw_data",
    "Test Data": "server/object_detection_classification/dataset/test",
    "Train Data": "server/object_detection_classification/dataset/train",
    "Validation Data": "server/object_detection_classification/dataset/valid"
}

def count_subfolders(folder_path):
    """
    Count the number of subfolders in a directory.
    """
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    return len(subfolders)

def count_images_in_subfolders(folder_path):
    """
    Count the number of images in each subfolder of a directory.
    """
    subfolder_counts = {}
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            image_count = len([name for name in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, name))])
            subfolder_counts[subfolder] = image_count
    return subfolder_counts

# Function to write the number of subfolders and images for each directory into a text file
def write_file(dir_paths):
    output_file = "server/object_detection_classification/dataset/img_counts.txt"
    with open(output_file, 'w') as file:
        for dir_name, dir_path in dir_paths.items():
            file.write(f"Directory: {dir_name}\n")
            file.write(f"Number of subfolders: {count_subfolders(dir_path)}\n")
            subfolder_counts = count_images_in_subfolders(dir_path)
            for subfolder, count in subfolder_counts.items():
                file.write(f"{subfolder}: {count} images\n")
            file.write("\n")

# Function to plot the counts of images in subdirectories for each directory
def plot_image_counts(dir_paths):
    num_dirs = len(dir_paths)
    fig, axs = plt.subplots(num_dirs, 1, figsize=(10, num_dirs * 5), sharex=True)

    for i, (dir_name, dir_path) in enumerate(dir_paths.items()):
        subfolder_counts = count_images_in_subfolders(dir_path)
        axs[i].bar(subfolder_counts.keys(), subfolder_counts.values(), color='skyblue')
        axs[i].set_title(f"Directory: {dir_name}")
        axs[i].set_ylabel('Number of Images')
        axs[i].tick_params(axis='x', rotation=45)

    plt.xlabel('Subdirectories (Classes)')
    plt.tight_layout()
    plt.show()
    plt.savefig('server/object_detection_classification/dataset/img_counts.png')

# Function to count subfolders and images in each subfolder for each directory
def analyse_directories(dir_paths):
    for dir_name, dir_path in dir_paths.items():
        print(f"Directory: {dir_name}")
        print(f"Number of subfolders: {count_subfolders(dir_path)}")
        subfolder_counts = count_images_in_subfolders(dir_path)
        for subfolder, count in subfolder_counts.items():
            print(f"{subfolder}: {count} images")
        print("\n")
    #write_file(dir_paths)
    #plot_image_counts(dir_paths)

# show directories
analyse_directories(DIR_PATHS)

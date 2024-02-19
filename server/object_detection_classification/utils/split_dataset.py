import glob, os, shutil
import numpy as np
import pandas as pd

def split_dataset(data_dir, train_dir, valid_dir, split_ratio): 
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(valid_dir):
        os.mkdir(valid_dir)
    num_training_files = 0
    num_validation_files = 0

    for subdir, dirs, files in os.walk(data_dir):
        category_name = os.path.basename(subdir)
        if category_name == os.path.basename(data_dir):
            continue
        training_data_category_dir = os.path.join(train_dir, category_name)
        validation_data_category_dir = os.path.join(valid_dir, category_name)
        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)
        if not os.path.exists(validation_data_category_dir):
            os.mkdir(validation_data_category_dir)
        file_list = glob.glob(os.path.join(subdir,'*.jpg'))
        print(str(category_name) + ' has ' + str(len(files)) + ' images') 
        random_set = np.random.permutation((file_list))
        # copy percentage of data from each category to train and test directory
        train_list = random_set[:round(len(random_set)*(split_ratio))] 
        test_list = random_set[-round(len(random_set)*(1-split_ratio)):]
        for lists in train_list : 
            shutil.copy(lists, train_dir + '/' + category_name + '/' )
            num_training_files += 1
        for lists in test_list : 
            shutil.copy(lists, valid_dir + '/' + category_name + '/' )
            num_validation_files += 1
    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_validation_files) + " testing files.")

def create_classes_csv(directory):
    # Specify the target CSV file path
    csv_path = os.path.join(directory, '_classes.csv')

    # List all subdirectories in the dataset path
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]

    # Create an empty DataFrame with the 'filename' and class columns
    df = pd.DataFrame(columns=['filename'] + subdirectories)

    # Loop through subdirectories
    for subfolder in subdirectories:
        subfolder_path = os.path.join(directory, subfolder)
        
        # List image files in the subdirectory with full paths
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Update the 'filename' column with image file names
        df = pd.concat([df, pd.DataFrame({'filename': image_files})], ignore_index=True)

        # Set the specific class column to 1 for each image
        df.loc[df['filename'].isin(image_files), subfolder] = 1

    # Fill NaN values with 0
    df = df.fillna(0)

    # Write the DataFrame to the _classes.csv file
    df.to_csv(csv_path, index=False)
    print(f"Created _classes.csv file: {csv_path}")


# Directory Paths
data_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/dataset/raw_data"
train_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/dataset/train"
valid_dir = "C:/Users/miku/Documents/Yew Jin/FYP_20297501/server/object_detection/dataset/valid"

split_ratio = 0.8

print("1. split dataset\n2. create csv")
selection = input("Selection: ")
if selection == '1':
    split_dataset(data_dir, train_dir, valid_dir, split_ratio)
elif selection == '2':
    for directory in [train_dir, valid_dir]:
        create_classes_csv(directory)
print("Done!")

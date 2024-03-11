import glob, os, shutil
import numpy as np
import pandas as pd

def split_dataset(data_dir, train_dir, valid_dir, test_dir, split_ratios=(0.7, 0.2, 0.1)):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    num_training_files = 0
    num_validation_files = 0
    num_test_files = 0

    for subdir, dirs, files in os.walk(data_dir):
        category_name = os.path.basename(subdir)
        if category_name == os.path.basename(data_dir):
            continue

        training_data_category_dir = os.path.join(train_dir, category_name)
        validation_data_category_dir = os.path.join(valid_dir, category_name)
        test_data_category_dir = os.path.join(test_dir, category_name)
        
        if not os.path.exists(training_data_category_dir):
            os.makedirs(training_data_category_dir)
        if not os.path.exists(validation_data_category_dir):
            os.makedirs(validation_data_category_dir)
        if not os.path.exists(test_data_category_dir):
            os.makedirs(test_data_category_dir)
        
        file_list = glob.glob(os.path.join(subdir, '*.jpg'))
        print(str(category_name) + ' has ' + str(len(file_list)) + ' images')
        
        random_set = np.random.permutation(file_list)
        num_files = len(random_set)
        
        train_end = int(num_files * split_ratios[0])
        valid_end = int(num_files * (split_ratios[0] + split_ratios[1]))

        train_list = random_set[:train_end]
        valid_list = random_set[train_end:valid_end]
        test_list = random_set[valid_end:]

        for lists in train_list:
            shutil.copy(lists, os.path.join(train_dir, category_name))
            num_training_files += 1
        for lists in valid_list:
            shutil.copy(lists, os.path.join(valid_dir, category_name))
            num_validation_files += 1
        for lists in test_list:
            shutil.copy(lists, os.path.join(test_dir, category_name))
            num_test_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_validation_files) + " validation files.")
    print("Processed " + str(num_test_files) + " test files.")

def create_classes_csv(directory):
    csv_path = os.path.join(directory, '_classes.csv')
    subdirectories = [subdir for subdir in os.listdir(directory) if os.path.isdir(os.path.join(directory, subdir))]
    df = pd.DataFrame(columns=['filename'] + subdirectories)
    
    for subfolder in subdirectories:
        subfolder_path = os.path.join(directory, subfolder)
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        df = pd.concat([df, pd.DataFrame({'filename': image_files})], ignore_index=True)
        df.loc[df['filename'].isin(image_files), subfolder] = 1
    
    df = df.fillna(0)
    df.to_csv(csv_path, index=False)
    print(f"Created _classes.csv file: {csv_path}")

# Directory Paths
data_dir = "server/object_detection_classification/dataset/raw_data"
train_dir = "server/object_detection_classification/dataset/train"
valid_dir = "server/object_detection_classification/dataset/valid"
test_dir = "server/object_detection_classification/dataset/test"

split_ratios = (0.7, 0.1, 0.2)

print("1. Split dataset\n2. Create CSV")
selection = input("Selection: ")
if selection == '1':
    split_dataset(data_dir, train_dir, valid_dir, test_dir, split_ratios)
elif selection == '2':
    for directory in [train_dir, valid_dir, test_dir]:
        create_classes_csv(directory)
print("Done!")
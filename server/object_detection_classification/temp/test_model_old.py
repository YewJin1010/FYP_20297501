import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

def read_csv(file_path):
    return pd.read_csv(file_path)

def extract_columns(dataframe):
    return dataframe.keys().values.tolist()

def extract_subfolder(filename):
    parts = filename.split('_')
    if len(parts) == 2 and parts[-1].isdigit():
        return parts[0]
    else:
        return '_'.join(parts[:-1])

def create_data_generators(directory, csv_filename, columns):
    df = pd.read_csv(os.path.join(directory, csv_filename))
    df['filename'] = df['filename'].str.strip()
    df['full_path'] = df['filename'].apply(lambda x: os.path.join(directory, x))
    df['subfolder'] = df['filename'].apply(extract_subfolder)
    df['full_path'] = df.apply(lambda row: os.path.join(directory, row['subfolder'], row['filename']), axis=1)
    
    invalid_images = []

    for index, row in df.iterrows():
        if not os.path.exists(row['full_path']):
            invalid_images.append(row['filename'])
            print(f"Invalid image: {row['filename']}, Full path: {row['full_path']}")

    data_generator = ImageDataGenerator()

    data_flow = data_generator.flow_from_dataframe(
        dataframe=df,
        directory='',
        x_col="full_path",
        y_col=columns,
        batch_size=32,
        class_mode=None,
        target_size=(224, 224),
        shuffle=False
    )
    return df, data_flow

def test_on_testing_set(test_dir, model_filename, columns):
    
    _, test_generator = create_data_generators(test_dir, "_classes.csv", columns)

    model = load_model(model_filename)
    
    results = []

    for i in range(len(test_generator)):
        batch_images, _ = test_generator[i]
        batch_size = len(batch_images)

        # Make predictions for each image in the batch
        batch_pred = model.predict(batch_images)

        # Convert predicted probabilities to binary values
        batch_pred_bool = (batch_pred > 0.5).astype(int)

        # Create DataFrame with predicted class labels and filenames for the batch
        batch_results = pd.DataFrame(batch_pred_bool, columns=columns)
        batch_results["filename"] = [os.path.basename(file) for file in test_generator.filepaths[i * test_generator.batch_size : (i+1) * test_generator.batch_size]]
        ordered_cols = ["filename"] + columns
        batch_results = batch_results[ordered_cols]

        # Append batch results to the overall results list
        results.append(batch_results)

    # Concatenate results from all batches
    results = pd.concat(results, ignore_index=True)

    # Save results to a CSV file
    results.to_csv("C:/Users/yewji/FYP_20297501/server/object_detection/results/results.csv", index=False)
    print("Results saved to results.csv")

def compare_results(results_csv_path, test_csv_path):
        # Read CSV files
    results_df = pd.read_csv(results_csv_path)
    test_df = pd.read_csv(test_csv_path)

    # Extract Columns
    train_columns = extract_columns(train_data)

    # Remove filename column
    columns = [col for col in train_columns if col != 'filename']

    # Initialize counters
    correct_count = 0
    total_count = 0

    # Iterate through each row in the results DataFrame
    for index, result_row in results_df.iterrows():
        # Find the corresponding row in the test DataFrame based on 'filename'
        test_row = test_df[test_df['filename'] == result_row['filename']].iloc[0]

        # Compare the values in each row (excluding the 'filename' column)
        values_match = (result_row[columns] == test_row[columns]).all()

        # Update counters
        correct_count += values_match
        total_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return accuracy*100

def test_on_custom_images(model_filename, image_paths, columns):

    model = load_model(model_filename)

    # Create a temporary data generator for the custom images
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    custom_generator = data_generator.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': image_paths}),
        directory='',  # Since absolute paths are provided
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    predict = model.predict_generator(custom_generator, steps = len(image_paths))
    y_classes = predict.argmax(axis=-1)
    print(y_classes)
   
    # Assuming 'test_generator' is an instance of the test data generator
    image_names = [os.path.basename(filename) for filename in custom_generator.filenames]

    # Map class numbers to class names
    predicted_class_names = [columns[class_num] for class_num in y_classes]

    # Combine image names with predicted classes and class names
    results = zip(image_names, y_classes, predicted_class_names)

    # Set the output directory
    output_directory = "C:/Users/yewji/FYP_20297501/server/object_detection/results"
    
    # Write to a text file in the specified directory
    output_file = os.path.join(output_directory, "predicted_results_custom_images.txt")
    with open(output_file, "w") as file:
        for image_name, predicted_class, predicted_class_name in results:
            file.write(f"{image_name}: Predicted Class {predicted_class} ({predicted_class_name})\n")

    print(f"Predicted results with class names written to: {output_file}")

    """
    # Make predictions for each custom image
    for i in range(len(custom_generator)):
        batch_images = custom_generator[i]
        predictions.append(model.predict(batch_images))

   # Decode and print the top-3 predicted classes
    decoded_predictions = [(columns[i], score) for i, score in enumerate(predictions[0])]
    decoded_predictions = sorted(decoded_predictions, key=lambda x: x[1], reverse=True)[:3]

    print("Predictions:")
    for i, (label, score) in enumerate(decoded_predictions):
        # Extract the score value from the NumPy array or list
        score_value = score.flatten().tolist()[0] if isinstance(score, np.ndarray) else score[0]
        print(f"{i + 1}: {label} ({score_value:.2f})")
    """

def test_on_single_image(model, image_path, class_names):
    
    model = load_model(model_filename)

    # Create a temporary data generator for the custom image
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    image_generator = data_generator.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': [image_path]}),
        directory='',  # Since absolute paths are provided
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size= (224, 224),
        batch_size=1,
        shuffle=False
    )

    # Make predictions for the custom image
    predictions = model.predict(image_generator[0])

    # Generate argmax for predictions
    class_id = np.argmax(predictions, axis=1)

    # Transform class number into class name
    class_name = class_names[class_id.item()]

    return class_name, predictions.flatten().tolist()

def test_on_testing_set2(test_dir, model_filename, columns):

    model = load_model(model_filename)

    _, test_generator = create_data_generators(test_dir, "_classes.csv", columns)    
    
    predict = model.predict_generator(test_generator, steps=len(test_generator.filenames))
    y_classes = predict.argmax(axis=-1)

    # Assuming 'test_generator' is an instance of the test data generator
    image_names = [os.path.basename(filename) for filename in test_generator.filenames]

    # Map class numbers to class names
    predicted_class_names = [columns[class_num] for class_num in y_classes]

    # Combine image names with predicted classes and class names
    results = zip(image_names, y_classes, predicted_class_names)

    # Set the output directory
    output_directory = "C:/Users/yewji/FYP_20297501/server/object_detection/results"
    
    # Write to a text file in the specified directory
    output_file = os.path.join(output_directory, "predicted_results_test_set.txt")
    with open(output_file, "w") as file:
        for image_name, predicted_class, predicted_class_name in results:
            file.write(f"{image_name}: Predicted Class {predicted_class} ({predicted_class_name})\n")

    print(f"Predicted results with class names written to: {output_file}")

def test_on_testing_set_3(test_dir,model_filename, columns ):
    model = load_model(model_filename)

    # Create an ImageDataGenerator
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

    
    test_generator = data_generator.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,  # Do not use class labels, as you are doing prediction
        shuffle=False
    )
    predict = model.predict_generator(test_generator, steps=len(test_generator.filenames))
    y_classes = predict.argmax(axis=-1)

    # Assuming 'test_generator' is an instance of the test data generator
    image_names = [os.path.basename(filename) for filename in test_generator.filenames]

    # Map class numbers to class names
    predicted_class_names = [columns[class_num] for class_num in y_classes]

    # Combine image names with predicted classes and class names
    results = zip(image_names, y_classes, predicted_class_names)

    # Set the output directory
    output_directory = "C:/Users/yewji/FYP_20297501/server/object_detection/results"
    
    # Write to a text file in the specified directory
    output_file = os.path.join(output_directory, "predicted_results_test_set_3.txt")
    with open(output_file, "w") as file:
        for image_name, predicted_class, predicted_class_name in results:
            file.write(f"{image_name}: Predicted Class {predicted_class} ({predicted_class_name})\n")

    print(f"Predicted results with class names written to: {output_file}")

# Model Directory
model_path = "C:/Users/yewji/FYP_20297501/server/object_detection/models"
model_filename = "C:/Users/yewji/FYP_20297501/server/object_detection/resnet50_testing/resnet50_categorical_model.h5"

# Read CSVs
train_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/train/_classes.csv')
test_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/test/_classes.csv')
valid_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/valid/_classes.csv')

# Directory Paths
train_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/train"
test_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/test"
valid_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/valid"

# Plot Path
plot_path = "C:/Users/yewji/FYP_20297501/server/object_detection/results"

# Result CSV
results_csv_path = "C:/Users/yewji/FYP_20297501/server/object_detection/results/results.csv"
test_csv_path = "C:/Users/yewji/FYP_20297501/server/object_detection/test/_classes.csv"

# Image Paths
image_path_1 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/blueberries.jpg"
image_path_2 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/eggs.jpg"

print("1. Test on test set")
print("2. Compare results for test set")
print("3. Test on custom images")
print("4. Test on single image")
print("5. Test on testing set 2")
print("6. Test on testing set 3")
choice = input("Enter your choice (1/2/3/4/5/6): ")

# Extract Columns
train_columns = extract_columns(train_data)

# Remove filename column
columns = [col for col in train_columns if col != 'filename']

if choice == "1":
    test_on_testing_set(test_dir, model_filename, columns)
    print("Would you like to compare results?")
    choice = input("Enter your choice (y/n): ")

    if choice == 'y': 
        accuracy = compare_results(results_csv_path, test_csv_path)
        print("Accuracy:", accuracy)
                
    elif choice == 'n':
        print("No comparison selected.")
    else:
        print("Invalid choice. Please enter a valid option (y/n).")
    
elif choice == "2":
    accuracy = compare_results(results_csv_path, test_csv_path)
    print("Accuracy:", accuracy)

elif choice == "3":
    image_paths = [image_path_1, image_path_2]
    test_on_custom_images(model_filename, image_paths, columns)

elif choice == "4":
    class_name, predictions = test_on_single_image(model_filename, image_path_2, columns)
    print("Predicted class name:", class_name)
    print("Predicted class probabilities:", predictions)

elif choice == "5":
    test_on_testing_set2(test_dir,model_filename, columns)

elif choice == "6":
    test_on_testing_set_3(test_dir, model_filename, columns)
else: 
    print("Invalid choice. Please enter a valid option (1/2/3/4/5).")
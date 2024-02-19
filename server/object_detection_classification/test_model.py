import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_csv(file_path):
    return pd.read_csv(file_path)

def extract_columns(dataframe):
    return dataframe.keys().values.tolist()

def test_on_testing_set(test_dir, model, columns):
    model = load_model(model)
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
    image_names = [os.path.basename(filename) for filename in test_generator.filenames]

    # Map class numbers to class names
    predicted_class_names = [columns[class_num] for class_num in y_classes]

    # Create a DataFrame with columns 'filename' and class names
    df_results = pd.DataFrame(columns=['filename'] + columns)

    # Set the 'filename' column
    df_results['filename'] = image_names

    # Set the values in the DataFrame based on predictions
    for class_name in columns:
        class_indices = [i for i, predicted_class in enumerate(predicted_class_names) if predicted_class == class_name]
        df_results[class_name].iloc[class_indices] = 1

    # Fill NaN values with 0
    df_results = df_results.fillna(0)

    # Save the DataFrame to a CSV file
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}")

def compare_results(results_csv_path, test_data):
        # Read CSV files
    results_df = pd.read_csv(results_csv_path)

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
        test_row = test_data[test_data['filename'] == result_row['filename']].iloc[0]

        # Compare the values in each row (excluding the 'filename' column)
        values_match = (result_row[columns] == test_row[columns]).all()

        # Update counters
        correct_count += values_match
        total_count += 1

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return accuracy*100

def test_on_sample_images(model, image_paths, columns, results_path):

    model = load_model(model)

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
   
    fig, axs = plt.subplots(len(image_paths), 2, figsize=(12, 6 * len(image_paths)))

    for i, (image_path, prediction) in enumerate(zip(image_paths, predict)):
        image_name = os.path.basename(image_path)

        # Get the top 5 class indices with highest confidence
        top5_indices = np.argsort(prediction)[::-1][:5]
        print(prediction[top5_indices])

        # Load and plot the image to the left of the bar chart
        img = mpimg.imread(image_path)

        # Plot the image
        axs[i, 0].imshow(img)
        axs[i, 0].axis('off')
        axs[i, 0].set_title(f'Image: {image_name}')

        # Plot the vertical bar chart to the right
        axs[i, 1].bar(range(5), prediction[top5_indices], align='center', alpha=0.7)
        axs[i, 1].set_xticks(range(5))
        axs[i, 1].set_xticklabels([columns[idx] for idx in top5_indices], rotation=45, ha='right')
        axs[i, 1].set_ylabel('Confidence')
        axs[i, 1].set_title('Top 5 Predictions')

    plt.subplots_adjust(hspace=0.5)

    # Save the combined plot
    output_file_path = os.path.join(results_path, "sample_images_prediction_plots.png")
    plt.savefig(output_file_path)
    plt.show()

    print(f"Combined prediction plot saved to: {output_file_path}")

# Model Directory
model = "C:/Users/yewji/FYP_20297501/server/object_detection/trained_models/resnet50.h5"

# Read CSVs
train_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/dataset/train/_classes.csv')
valid_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/dataset/valid/_classes.csv')
test_data = read_csv('C:/Users/yewji/FYP_20297501/server/object_detection/dataset/test/_classes.csv')

# Directory Paths
data_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/train"
train_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/dataset/train"
valid_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/dataset/valid"
test_dir = "C:/Users/yewji/FYP_20297501/server/object_detection/dataset/test"

# Result CSV
results_csv_path = "C:/Users/yewji/FYP_20297501/server/object_detection/results/test_results/results.csv"
results_path = "C:/Users/yewji/FYP_20297501/server/object_detection/results/test_results"

# Sample Image Paths
image_path_1 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/blueberries.jpg"
image_path_2 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/eggs.jpg"
image_path_3 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/flour.jpg"
image_path_4 = "C:/Users/yewji/FYP_20297501/server/object_detection/sample_images/apple+banana.jpeg"

print("1. Test on test set")
print("2. Compare results with test csv")
print("3. Test on sample images")
choice = input("Enter your choice (1/2/3): ")

# Extract Columns
train_columns = extract_columns(train_data)

# Remove filename column
columns = [col for col in train_columns if col != 'filename']
print(columns)

if choice == "1":
    test_on_testing_set(test_dir, model, columns)
elif choice == "2":
    accuracy = compare_results(results_csv_path, test_data)
    print(f"Accuracy: {accuracy:.2f}%")
elif choice == "3":
    image_paths = [image_path_1, image_path_2, image_path_3, image_path_4]
    test_on_sample_images(model, image_paths, columns, results_path)
else: 
    print("Invalid choice. Please enter a valid option (1/2/3)")

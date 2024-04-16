import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.imagenet_utils import preprocess_input, decode_predictions 
from keras.preprocessing.image import ImageDataGenerator
import datetime
from sklearn.metrics import confusion_matrix

def test_on_testing_set(test_dir, model, classes, results_path):

    model = load_model(model, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = data_generator.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,  
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

    now = datetime.datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")

    # Save the DataFrame to a CSV file
    save_path = os.path.join(results_path, f'results_{date_time}.csv')
    df_results.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")
    return df_results
    
def create_confusion_matrix(df_results, test_data, results_path, classes):

    # Get the actual values from the test data
    actual_values = test_data[classes].values

    # Get the predicted values from the results DataFrame
    predicted_values = df_results[classes].values

    # Create the confusion matrix
    cm = confusion_matrix(actual_values.argmax(axis=1), predicted_values.argmax(axis=1))

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=columns, yticklabels=columns)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    #plt.savefig(os.path.join(results_path, 'confusion_matrix.png'))
    plt.show()

# Model Directory
model = "server/object_detection_classification/trained_models/resnet50.h5"

# Read CSVs
test_data = pd.read_csv('server/object_detection_classification/dataset/test/_classes.csv')

# Directory Paths
test_dir = "server/object_detection_classification/dataset/test"

# Result CSV
results_path = "server/object_detection_classification/results/test_results"

# Classes 
columns = test_data.keys().values.tolist()
classes = [col for col in columns if col != 'filename']

df_results = test_on_testing_set(test_dir, model, classes, results_path)

create_confusion_matrix(df_results, test_data, results_path, classes)


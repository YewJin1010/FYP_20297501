from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

def test_on_testing_set(test_data, model, columns):
    # Load the model
    model = load_model(model)
        
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate through test data and generate predictions
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = data_generator.flow_from_directory(
        test_data,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,  # Do not use class labels, as you are doing prediction
        shuffle=False
    )
    for i in range(len(test_generator.filenames)):
        # Generate prediction for each image
        img = next(test_generator)
        prediction = model.predict(img)
        
        # Get true label from test data
        true_label = test_data.loc[test_data['filename'] == os.path.basename(test_generator.filenames[i]), 'true_label'].values[0]
        
        # Map predicted label to class name
        predicted_label = columns[np.argmax(prediction)]
        
        # Append true label and predicted label to lists
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Return true labels and predicted labels
    return true_labels, predicted_labels

def create_confusion_matrix(true_labels, predicted_labels, classes):
    # Generate confusion matrix using scikit-learn
    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=classes)
    return confusion_mat

# Load test data
test_data = pd.read_csv('server/object_detection_classification/dataset/test/_classes.csv')
# Load model
model = "server/object_detection_classification/trained_models/resnet50.h5"

# Extract columns from test data
columns = test_data.keys().values.tolist()
classes = [col for col in columns if col != 'filename']


true_labels, predicted_labels = test_on_testing_set(test_data, model, columns)
confusion_mat = create_confusion_matrix(true_labels, predicted_labels, classes)
print("Confusion Matrix:")
print(confusion_mat)

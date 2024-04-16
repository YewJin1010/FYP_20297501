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


def test_model(model, image_paths, columns, results_path):

    model = load_model(model, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Create a temporary data generator for the custom images
    data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
    custom_generator = data_generator.flow_from_dataframe(
        dataframe=pd.DataFrame({'filename': image_paths}),
        directory='', 
        x_col='filename',
        y_col=None,
        class_mode=None,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False
    )

    # Predict the custom images
    predict = model.predict(custom_generator, steps = len(image_paths))

    # Plot the images and predictions
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

    now = datetime.datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    #
    # Save the combined plot
    output_file_path = os.path.join(results_path, f'predictions_{date_time}.png')
    plt.savefig(output_file_path)
    plt.show()

    print(f"Combined prediction plot saved to: {output_file_path}")

# Model Directory
model = "server/object_detection_classification/trained_models/resnet50.h5"
# Test directory
test_dir = "server/object_detection_classification/dataset/test"
# Test data csv
test_data = pd.read_csv('server/object_detection_classification/dataset/test/_classes.csv')
# Results path 
results_path = "server/object_detection_classification/results/test_results"

# Classes 
columns = test_data.keys().values.tolist()
classes = [col for col in columns if col != 'filename']

# Sample Image Paths
images = ['blueberries', 'eggs', 'flour', 'carrots', 'fruits']
image_paths = []

for image in images: 
    image_path = f'server/object_detection_classification/sample_images/{image}.jpg'
    image_paths.append(image_path)

test_model(model, image_paths, columns, results_path)
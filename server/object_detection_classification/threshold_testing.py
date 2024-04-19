import time, os, cv2, glob
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import tensorflow as tf
from keras.applications import ResNet50 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions 
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to the saved model
PATH_TO_SAVED_MODEL = 'server/object_detection_classification/tensorflow/pretrained_models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
# Path to the label map
PATH_TO_LABELS = 'server/object_detection_classification/tensorflow/data/mscoco_complete_label_map.pbtxt'

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Model loaded after {} seconds'.format(elapsed_time))

# Label Map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Path to ResNet50 model
model = "server/object_detection_classification/trained_models/resnet50.h5"

def get_class_list():
    dataframe = pd.read_csv('server/object_detection_classification/dataset/train/_classes.csv')
    columns = dataframe.keys().values.tolist()
    class_list = [col for col in columns if col != 'filename']
    return class_list

def plot_detections(image_np_with_detections, filtered_detections, image_name, detection_score_threshold):
    for i in range(filtered_detections['detection_boxes'].shape[0]):
        ymin, xmin, ymax, xmax = filtered_detections['detection_boxes'][i]
        xmin = int(xmin * image_np_with_detections.shape[1])
        xmax = int(xmax * image_np_with_detections.shape[1])
        ymin = int(ymin * image_np_with_detections.shape[0])
        ymax = int(ymax * image_np_with_detections.shape[0])
        # Draw bounding box
        cv2.rectangle(image_np_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    plt.imshow(image_np_with_detections)
    plt.axis('off')
    save_path = f'server/object_detection_classification/results/detection_threshold_test_results'
    # Create the directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f'{save_path}/{image_name}_threshold_{detection_score_threshold}.jpg')

def preprocess_image(image_path):
    print("Preprocessing image...")
    # Resize the image 
    resized_image = cv2.resize(image_path, (224, 224))
    # Convert the image to RGB color format
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    # Convert the image to a TensorFlow tensor
    tensor_image = tf.convert_to_tensor(rgb_image, dtype=tf.uint8)
    # Add an extra dimension to represent batch size
    tensor_image = tf.expand_dims(tensor_image, axis=0)
    return tensor_image
 
def detect_objects(image_np, detection_score_threshold, image_name, plot_results):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    # Perform inference
    try: 
        detections = detect_fn(input_tensor)
    except Exception as e:
        print(f"Error: {e}")
        preprocessed_image = preprocess_image(image_np)
        detections = detect_fn(preprocessed_image)
        print("Error resolved!")

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()

    # Filter out detections below the threshold
    above_threshold = detections['detection_scores'] > detection_score_threshold
    filtered_detections = {key: value[above_threshold]
        for key, value in detections.items() if isinstance(value, np.ndarray)}
    
    if plot_results:
        plot_detections(image_np_with_detections, filtered_detections, image_name, detection_score_threshold)

    print('\nDetection finished!')
    return image_np_with_detections, filtered_detections

def extract_and_resize_rois(image_np_with_detections, filtered_detections):
    rois_list = []
    print("Number of detections:", len(filtered_detections['detection_boxes']))
    image_height, image_width, _ = image_np_with_detections.shape
    boxes = filtered_detections['detection_boxes']
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        ymin = int(ymin * image_height)
        xmin = int(xmin * image_width)
        ymax = int(ymax * image_height)
        xmax = int(xmax * image_width)
        roi = image_np_with_detections[ymin:ymax, xmin:xmax]
        resized_roi = cv2.resize(roi, (224, 224))  # Resize to fit ResNet50 input size
        rois_list.append((resized_roi, [xmin, ymin, xmax, ymax]))
    print("ROIs extracted and resized!")
    return rois_list

def save_roi_image(roi_image, image_name, i, result, classification_score_threshold):
    roi_image_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    save_path = f'server/object_detection_classification/results/roi_images/{image_name}/{classification_score_threshold}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_path = os.path.join(save_path, f"roi_{result}_{i}.jpg")
    cv2.imwrite(image_path, roi_image_rgb)
    print(f"ROI image saved: {image_path}")

# Function to classify ROIs using a pre-trained ResNet50 model
def classify_rois(model, rois_list, class_list, classification_score_threshold, image_name):
    # Load pre-trained ResNet50 model
    model = load_model(model, compile=False)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    results = []
    roi_boxes = []  # List to store the bounding box coordinates for each ROI
    try:
        for i, (roi, roi_box) in enumerate(rois_list):
            try:
                print("Attempting approach 1")
                preprocessed_roi = preprocess_input(np.array([roi]))  # Preprocess a single ROI
            except Exception as e:
                print(f"Error: {e}")
                print("Attempting approach 2")
                # Resize ROI to (224, 224) dimensions
                resized_roi = cv2.resize(roi, (224, 224))
                # Preprocess resized ROI
                preprocessed_roi = preprocess_input(np.array([resized_roi]))

            # Predict class probabilities for the ROI
            predictions = model.predict(preprocessed_roi)

            top_indices = np.argsort(predictions)[0, ::-1][:5]

            top_predictions = [(class_list[idx], predictions[0, idx]) for idx in top_indices]
            print(f"Top predictions for ROI {i}: {top_predictions}")

            for j, prediction in enumerate(predictions):
                # Get top 5 class indices with highest confidence
                top_index = prediction.argmax()  # Get the index of the class with the highest probability
                top_class = class_list[top_index]
                top_score = prediction[top_index]  # Retrieve the score corresponding to the top class
                if top_score >= classification_score_threshold:
                    result = [(top_class, top_score)]
                    results.append(result)
                    # Save the ROI image
                    save_roi_image(roi, image_name, f"{i}_{j}", result, classification_score_threshold)

    except Exception as e:
        print(f"Error: {e}")

    print("ROIs classified!")
    print("\nclassification_results: ", results)
    return results, roi_boxes

# Function to combine detection and classification results
def combine_results(classification_results, roi_boxes):
    combined_results = []
    num_detections = len(classification_results)
    print(f"Processing image with {num_detections} detections")
    for j, (prediction, box) in enumerate(zip(classification_results, roi_boxes)):
        print(f"Processing detection {j+1}/{num_detections}")
        class_label, class_score = prediction[0]  # Extract class label and score from tuple
        ymin, xmin, ymax, xmax = box  # Extract bounding box coordinates
        result = {
            'class_label': class_label,
            'class_score': class_score,
            'bounding_box': [xmin, ymin, xmax, ymax],
        }
        combined_results.append(result)
    print("Results combined!")
    return combined_results

def detect_and_classify(image_path, image_name, detection_score_threshold, classification_score_threshold, plot_detections):
    print('Running inference for {}... '.format(image_path), end='')
    # Load image into numpy array
    image_np = np.array(Image.open(image_path))
    image_np_with_detections, filtered_detections = detect_objects(image_np, detection_score_threshold, image_name, plot_detections)
    rois_list = extract_and_resize_rois(image_np_with_detections, filtered_detections)
    class_list = get_class_list()
    classification_results, roi_boxes = classify_rois(model, rois_list, class_list, classification_score_threshold, image_name)
    final_results = combine_results(classification_results, roi_boxes)
    
def detection_threshold_test(images):
    plot_results = input("Do you want to plot the detection results? (y/n): ")
    # Do not take inputs not y and n
    while plot_results not in ['y', 'n']:
        plot_results = input("Invalid input. Please enter 'y' or 'n': ")
    if plot_results == 'y':
        plot_results = True
    else:
        plot_results = False

    for image in images:
        image_path = f'server/object_detection_classification/sample_images/{image}.jpg'
        detection_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for threshold in detection_thresholds:
            print(f"Running inference for threshold: {threshold}")
            image_np = np.array(Image.open(image_path))
            image_np_with_detections, filtered_detections = detect_objects(image_np, threshold, image, plot_results)
            print("Inference finished!")

def classification_threshold_test(images):
    plot_detections = False

    for image in images:
        image_path = f'server/object_detection_classification/sample_images/{image}.jpg'
        detection_score_threshold = 0.2
        classification_score_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for threshold in classification_score_thresholds:
            print(f"Running inference for threshold: {threshold}")
            detect_and_classify(image_path, image, detection_score_threshold, threshold, plot_detections)
            print("Inference finished!")

images = ['apple_banana','fruits','bicycle', 'market']
# images = ['bicycle']
print("1. Detection threshold test")
print("2. Classification threshold test")

test_to_run = int(input("Enter the test to run: "))	
if test_to_run == 1:
    detection_threshold_test(images)
elif test_to_run == 2:
    classification_threshold_test(images)


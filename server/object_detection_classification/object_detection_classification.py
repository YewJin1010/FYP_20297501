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
PATH_TO_SAVED_MODEL = 'object_detection_classification/tensorflow/pretrained_models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8/saved_model'
# Path to the label map
PATH_TO_LABELS = 'object_detection_classification/tensorflow/data/mscoco_complete_label_map.pbtxt'

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
model = "object_detection_classification/trained_models/resnet50.h5"

def get_class_list():
    dataframe = pd.read_csv('object_detection_classification/dataset/train/_classes.csv')
    columns = dataframe.keys().values.tolist()
    class_list = [col for col in columns if col != 'filename']
    return class_list

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
 
# Function to detect objects in an image
def detect_objects(image_np, detection_score_threshold):

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
        rois_list.append(resized_roi)
    print("ROIs extracted and resized!")
    return rois_list

# Function to classify ROIs using a pre-trained ResNet50 model
def classify_rois(model, rois_list, class_list, classification_score_threshold):
    # Load pre-trained ResNet50 model
    model = load_model(model, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    results = []
    try:
        for rois in rois_list:
            try:
                print("Attempting approach 1")
                preprocessed_rois = preprocess_input(np.array(rois))
                preprocessed_rois = np.reshape(preprocessed_rois, (-1, 224, 224, 3))
            except Exception as e:
                print(f"Error: {e}")
                print("Attempting approach 2")
                # Initialize a list to store resized ROIs
                resized_rois_list = []

                # Resize each ROI to (224, 224) dimensions
                for roi in rois:
                    resized_roi = cv2.resize(roi, (224, 224))
                    resized_rois_list.append(resized_roi)

                # Stack resized ROIs into a single array
                resized_rois_array = np.stack(resized_rois_list)

                # Preprocess resized ROIs
                preprocessed_rois = preprocess_input(resized_rois_array)
                preprocessed_rois = np.stack([preprocessed_rois] * 3, axis=-1)

            # Predict class probabilities for all ROIs
            predictions = model.predict(preprocessed_rois)

            result = []
            for prediction in predictions:
                top_index = prediction.argmax()  # Get the index of the class with the highest probability
                top_class = class_list[top_index]
                top_score = prediction[top_index]  # Retrieve the score corresponding to the top class
                if top_score >= classification_score_threshold:
                    top_class = class_list[top_index]
                    result.append((top_class, top_score))
            # Append result if it is not empty
            if result:
                results.append(result)

    except Exception as e:
        print(f"Error: {e}")

    print("ROIs classified!")
    print("\nclassification_results: ", results)
    return results


# Function to combine detection and classification results
def combine_results(filtered_detections, classification_results):
    combined_results = []
    num_detections = len(filtered_detections['detection_boxes'])
    print(f"Processing image with {num_detections} detections")
    for j, (box, prediction) in enumerate(zip(filtered_detections['detection_boxes'], classification_results)):
        print(f"Processing detection {j+1}/{num_detections}")
        ymin, xmin, ymax, xmax = box
        class_label, class_score = prediction[0]  # Extract class label and score from tuple
        result = {
            'class_label': class_label,
            'class_score': class_score,
            'bounding_box': [xmin, ymin, xmax, ymax],
        }
        combined_results.append(result)
    print("Results combined!")
    return combined_results

# Function to draw bounding boxes on an image
def draw_boxes(final_results, image_path):
    image = plt.imread(image_path)
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image)
        
    # Iterate over each detection result
    for result in final_results:
        class_label = result['class_label']
        class_score = result['class_score']
        xmin, ymin, xmax, ymax = result['bounding_box']
        
        # Convert normalized coordinates to image coordinates
        height, width, _ = image.shape
        xmin *= width
        ymin *= height
        xmax *= width
        ymax *= height

        # Create a rectangle patch
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

        # Add label and score as text
        label_text = f'{class_label}: {class_score:.2f}'
        ax.text(xmin, ymin - 5, label_text, color='r')

    # Define the directory to save the plot
    plot_directory = os.path.join('object_detection_classification', 'results', 'inference_results')

    # Extract the file name from the FileStorage object
    image_name = image_path.filename

    # Construct the full path for saving the plot
    plot_path = os.path.join(plot_directory, image_name)

    # Save the plot
    plt.savefig(plot_path)
    # Show the plot
    print("Image with bounding boxes saved!")
    print("images saved at: ", plot_path)
    plt.show()

def detect_and_classify(image_path):
    # Confidence score threshold for detection
    detection_score_threshold = 0.2

    # Confidence score threshold for classification
    classification_score_threshold = 0.25

    print('Running inference for {}... '.format(image_path), end='')
    # Load image into numpy array
    image_np = np.array(Image.open(image_path))
    image_np_with_detections, filtered_detections = detect_objects(image_np, detection_score_threshold)
    rois_list = extract_and_resize_rois(image_np_with_detections, filtered_detections)
    class_list = get_class_list()
    classification_results = classify_rois(model, rois_list, class_list, classification_score_threshold)
    final_results = combine_results(filtered_detections, classification_results)
    draw_boxes(final_results, image_path)
    return final_results  



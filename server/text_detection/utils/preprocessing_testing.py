import cv2
import csv
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
from nltk.corpus import stopwords

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'server/text_detection/Tesseract-OCR/tesseract.exe'

# Function to preprocess the image
def preprocess_image(image):
    save_path = f'server/text_detection/results/preprocessing_images/{image_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_path, 'gray_image.jpg'), gray_image)

    # Resize the image
    resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC) # Resize the image
    cv2.imwrite(os.path.join(save_path, 'resized_image.jpg'), resized_image)

    # Apply gaussian blur to the image
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    cv2.imwrite(os.path.join(save_path, 'blurred_image.jpg'), blurred_image)

    # Apply laplacian filter to the image
    laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
    # Sharpen the image
    sharpened_image = np.uint8(np.clip(blurred_image - 0.5 * laplacian, 0, 255))
    cv2.imwrite(os.path.join(save_path, 'sharpened_image.jpg'), sharpened_image)

    # Increase brightness of the image
    sharpened_image = cv2.convertScaleAbs(sharpened_image, alpha=1.5, beta=0)
    cv2.imwrite(os.path.join(save_path, 'brightened_image.jpg'), sharpened_image)

    # Perform otsu thresholding to get binary image
    _, binary_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(save_path, 'binary_iamge.jpg'), binary_image)
    
    return binary_image


# Function to detect text in the image
def detect_text(image, confidence_threshold):
    detected_text_list = []

    # Get word-level bounding box coordinates, text, and confidence scores
    d = pytesseract.image_to_data(image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
   
    # Iterate through detected text regions
    for i in range(len(d['text'])):
        # Extract text, bounding box coordinates, and confidence score
        text = d['text'][i]
        left, top, width, height = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        confidence = d['conf'][i]
        # Append to detected text list
        detected_text_list.append((text, confidence, left, top, width, height))

    return detected_text_list

# Function to filter non-alphanumeric characters and stopwords from detected text
def filter_text(detected_text_list):
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    # Extract text from detected text list
    filtered_text = []

    for text_info in detected_text_list:
        text, _, _, _, _, _ = text_info
        text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
        text = ' '.join([word for word in text.lower().split() if word not in stop_words])  # Remove stopwords
        # Append to filtered text list
        if text:
            filtered_text.append(text)
    return filtered_text

# Function to draw bounding boxes and text labels on the image
def draw_boxes(image, detected_text_list):
    for text_info in detected_text_list:
        text, confidence, left, top, width, height = text_info
        # Draw bounding box
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
        # Add text label with confidence score
        label = f"{text} ({confidence}%)"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Load the ingredients from the CSV file
def get_ingredients():
    file_path = "server/text_detection/cleaned_ingredients.csv"
    ingredients = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            ingredients.extend(row)

    # Remove empty strings
    ingredients = list(filter(None, ingredients))
    return ingredients

# Function to perform text detection on the image
def get_text_detection(image_path, image_name):

    # Test if the image can be opened
    try: 
        image = Image.open(image_path)
    except Exception as e:
        error = str(e)
        print("Error reading the image: ", error)
        return error

    # Load the image into a numpy array
    image_np = np.array(image)
    # Read the image
    original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if original_image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(original_image)
        # Detect text in the image
        detected_text_list = detect_text(preprocessed_image)
        print("Detected text: ", detected_text_list)
        # Filter text
        filtered_text = filter_text(detected_text_list)
        print("Filtered text: ", filtered_text)
        # Do not draw if no text detected
        if len(detected_text_list) == 0:
            print("No text detected.")
            return []
        
        else: 
            # Draw bounding boxes and text labels
            draw_boxes(preprocessed_image, detected_text_list)

            # Display the image with detected text and bounding boxes
            plt.imshow(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            #plt.show()
            # Define the directory to save the plot
            plot_directory = f'server/text_detection/results/preprocessing_images/{image_name}'
            if not os.path.exists(plot_directory):
                os.makedirs(plot_directory)
            # Construct the full path for saving the plot
            plot_path = os.path.join(plot_directory, f'text_detection.jpg')
            # Save the plot
            plt.savefig(plot_path)
        
        # Get the list of ingredients
        ingredients_list = get_ingredients()
        # Check if any of the detected text is an ingredient
        detected_ingredients = [text for text in filtered_text if text in ingredients_list]
        print("Detected ingredients: ", detected_ingredients)
    else:
        print("Error reading the image.")
    
    return detected_ingredients


image_names = ['sugar.jpeg', 'flour.jpg', 'milk_powder.jpeg']
for image_name in image_names:
    image_path = f'server/text_detection/sample_images/{image_name}'
    detected_ingredients = get_text_detection(image_path, image_name)
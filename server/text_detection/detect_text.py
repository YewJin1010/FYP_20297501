import pytesseract 
import matplotlib.pyplot as plt
import numpy as np
import re, os, cv2, csv
from PIL import Image
import nltk
from nltk.corpus import stopwords

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/yewji/FYP_20297501/server/text_detection/Tesseract-OCR/tesseract.exe'

# Set the confidence threshold for text detection
confidence_threshold = 50

# Function to preprocess the image
def preprocess_image(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gaussian_blur_image = cv2.GaussianBlur(resized_image, (5, 5), 0)
    laplacian = cv2.Laplacian(gaussian_blur_image, cv2.CV_64F)
    sharpened_image = np.uint8(np.clip(gaussian_blur_image - 0.5 * laplacian, 0, 255))
   
    return sharpened_image

# Function to detect text in the image
def detect_text(image, confidence_threshold):
    detected_text_list = []

    # Get word-level bounding box coordinates, text, and confidence scores
    d = pytesseract.image_to_data(image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
   
    # Iterate through detected text regions
    for i in range(len(d['text'])):
        # Check confidence score against threshold
        if int(d['conf'][i]) > confidence_threshold:
            # Extract text, bounding box coordinates, and confidence score
            text = d['text'][i]
            left, top, width, height = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            confidence = d['conf'][i]
            # Append to detected text list
            detected_text_list.append((text, confidence, left, top, width, height))

    return detected_text_list

def filter_text(detected_text_list):
    text_list = []
    for text_info in detected_text_list:
        text, confidence, left, top, width, height = text_info
        text_list.append(text)
    
    # Remove non-alphanumeric characters
    text_list = [re.sub(r'\W+', ' ', text) for text in text_list]
    # Convert to lowercase
    text_list = [text.lower() for text in text_list]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text_list = [' '.join([word for word in text.split() if word not in stop_words]) for text in text_list]

    # Remove empty strings
    text_list = list(filter(None, text_list))

    return text_list

# Function to draw bounding boxes and text labels on the image
def draw_boxes(image, detected_text_list):
    for text_info in detected_text_list:
        text, confidence, left, top, width, height = text_info
        # Draw bounding box
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), 2)
        # Add text label with confidence score
        label = f"{text} ({confidence}%)"
        cv2.putText(image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def get_ingredients():
    file_path = "text_detection/sorted_ingredients.csv"
    ingredients = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        first_row_skipped = False
        for row in reader:
            if not first_row_skipped:
                first_row_skipped = True
                continue  # Skip the first row
            ingredients.extend(row)
    
    # Remove empty strings
    ingredients = list(filter(None, ingredients))
    return ingredients

def get_text_detection(image_path):
    # show image
    try: 
        image = Image.open(image_path)
    except Exception as e:
        error = str(e)
        print("here")
        return error

    plt.imshow(image)
    plt.savefig("text_detection/results/original_image.jpg")    

    # Load the image into a numpy array
    image_np = np.array(Image.open(image_path))
    # Read the image
    original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
   # Check if the image is read successfully
    if original_image is not None:
        # Preprocess the image
        preprocessed_image = preprocess_image(original_image)

        # Detect text in the image
        detected_text_list = detect_text(preprocessed_image, confidence_threshold)
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
            plot_directory = os.path.join('text_detection', 'results')

            # Extract the file name from the FileStorage object
            image_name = image_path.filename

            # Construct the full path for saving the plot
            plot_path = os.path.join(plot_directory, image_name)

            # Save the plot
            plt.savefig(plot_path)
        
        ingredients_list = get_ingredients()
        # Check if any of the detected text is an ingredient
        detected_ingredients = [text for text in filtered_text if text in ingredients_list]
        print("Detected ingredients: ", detected_ingredients)

    else:
        print("Error reading the image.")

    return detected_ingredients




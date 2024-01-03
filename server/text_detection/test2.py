# Loading the required python modules
import pytesseract # this is tesseract module
import matplotlib.pyplot as plt
import cv2 # this is opencv module
import glob
import os
from PIL import Image
from pytesseract import Output
from matplotlib.patches import Rectangle


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

"""
# Specify the path to the folder containing text images
path_for_images = os.path.join(os.getcwd(), "C:/Users/yewji/fyp/textdetection/images/")

# Create a dictionary to store actual text for each image file
actual_text_dict = {
    "license_plate.jpg": "8JP 698",
    "image2.jpg": "DEF456",
    # Add more entries for other image files and actual text
}

# List to store predicted text and accuracy results
predicted_text = []
accuracy_results = []

# Loop through each image in the folder
for image_path in glob.glob(os.path.join(path_for_images, "*.jpg")):
    # Get the image file name
    image_file_name = os.path.basename(image_path)
    
    # Get the actual text from the dictionary
    actual_text = actual_text_dict.get(image_file_name, "Unknown")
    
    # Open the image using Pillow (PIL)
    image = Image.open(image_path)
    image = image.convert("RGB")
    
    # Use pytesseract to perform OCR on the image
    predicted_result = pytesseract.image_to_string(image, lang='eng', config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Process the predicted result
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    predicted_text.append(filter_predicted_result)
    
    # Calculate accuracy
    num_matches = sum(a == p for a, p in zip(actual_text, filter_predicted_result))
    accuracy = round((num_matches / len(actual_text)) * 100, 2)
    accuracy_results.append(accuracy)

# Print the headers for the results
print("Image File Name", "\t", "Actual Text", "\t", "Predicted Text", "\t", "Accuracy (%)")
print("----------------", "\t", "-------------------", "\t", "-----------------------", "\t", "------------")

# Print results for each image
for image_file_name, actual_text, predict_text, accuracy in zip(actual_text_dict.keys(), actual_text_dict.values(), predicted_text, accuracy_results):
    print(image_file_name, "\t\t", actual_text, "\t\t\t", predict_text, "\t\t  ", accuracy)
"""
    
image_path = ("C:/Users/yewji/FYP_20297501/server/text_detection/images/vanillaessence.jpg")

# Read the image using cv2
image = cv2.imread(image_path)

# Check if image is read successfully
if image is not None:
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Grey scale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # Image resizing
    resized_image = cv2.resize(gray_image, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

    # Denoise image
    gaussian_blur_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    predicted_result = pytesseract.image_to_string(gaussian_blur_image, lang='eng',
    config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
    print("Detected text: ", filter_predicted_result)

    # Get word-level bounding box coordinates and text information
    d = pytesseract.image_to_data(gaussian_blur_image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])

   # Draw bounding boxes and text
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            # print("x:", x, "y:", y, "x2:", x2, "y2:", y2)
            print("x:", x, "y:", y, "w:", w, "h:", h)
            # Draw the bounding box on the scaled image
            cv2.rectangle(resized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the detected text next to the bounding box
            cv2.putText(resized_image, d['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
             # Print the detected text for the current bounding box
            print("Detected text for Box", i+1, ":", d['text'][i])

    # Show the scaled and annotated image
    plt.imshow(resized_image)
    plt.axis('off')
    plt.title('Regular Detection')
    plt.show()
        
    # Rotate the image by 90 degrees to make vertical text horizontal
    rotated_image = cv2.rotate(gaussian_blur_image, cv2.ROTATE_90_CLOCKWISE)

    # Apply OCR to the rotated image
    rotated_image_detected_text = pytesseract.image_to_string(rotated_image, lang='eng', config='--oem 3 --psm 6')

    print("Rotated detected text: ", rotated_image_detected_text)

    # Get word-level bounding box coordinates and text information
    d = pytesseract.image_to_data(rotated_image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])
        
   # Draw bounding boxes and text
    for i in range(n_boxes):
        if int(d['conf'][i]) > 0:
            x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
            # print("x:", x, "y:", y, "x2:", x2, "y2:", y2)
            print("x:", x, "y:", y, "w:", w, "h:", h)
            # Draw the bounding box on the scaled image
            cv2.rectangle(rotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the detected text next to the bounding box
            cv2.putText(rotated_image, d['text'][i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
             # Print the detected text for the current bounding box
            print("Detected text for Box", i+1, ":", d['text'][i])

    plt.imshow(rotated_image)
    plt.axis('off')
    plt.title('Rotated Detection')
    plt.show()

else:
    print("Error reading the image.")


"""
# Load the image
image_path = "C:/Users/yewji/fyp/textdetection/images/vanillaessence.jpg"
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use an edge detection technique like Canny to find text regions
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

# Find contours of text regions
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image for visualization
image_copy = image.copy()

# Loop through detected contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    text_region = gray_image[y:y+h, x:x+w]

    # Apply OCR to the text region
    detected_text = pytesseract.image_to_string(text_region, lang='eng', config='--oem 3 --psm 6')

    print("Detected text in region:")
    print(detected_text)

    # Draw a rectangle around the detected text region
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected text regions
cv2.imshow("Detected Text Regions", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
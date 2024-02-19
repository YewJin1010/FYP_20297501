import pytesseract 
from pytesseract import Output
import cv2 
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/yewji/FYP_20297501/server/text_detection/Tesseract-OCR/tesseract.exe'

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)
#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)
#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

def preprocess_image(image):
    gray = get_grayscale(image)
    blurred = remove_noise(gray)
    thresholded = thresholding(blurred)
    return thresholded

def detect_and_draw_text(img, texts_to_detect):
    preprocessed_img = preprocess_image(img)
    
    d = pytesseract.image_to_data(preprocessed_img, output_type=Output.DICT)
    n_boxes = len(d['text'])
    
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            detected_text = d['text'][i].lower()  # Convert to lowercase for case-insensitive matching
            for text_to_detect in texts_to_detect:
                if re.search(text_to_detect, detected_text):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    print("Detected text:", detected_text)

# Function to detect and print all the detected text
def detect_and_print_text(img):
    preprocessed_img = preprocess_image(img)
    
    d = pytesseract.image_to_data(preprocessed_img, output_type=Output.DICT)
    n_boxes = len(d['text'])
    
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            detected_text = d['text'][i].strip()  # Remove leading/trailing whitespaces
            print("Detected text:", detected_text)


# Define a list of texts to detect
texts_to_detect = ['invoice','due','date', 'total', 'your_custom_text_here']

# Read the image using cv2
image_path = "C:/Users/yewji/FYP_20297501/server/text_detection/images/invoice-sample.jpg"
img = cv2.imread(image_path)

# Detect and draw bounding boxes around specified texts
detect_and_draw_text(img, texts_to_detect)

detect_and_print_text(img)
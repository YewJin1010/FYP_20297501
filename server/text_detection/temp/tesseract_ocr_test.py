import pytesseract 
import matplotlib.pyplot as plt
import cv2
import glob
import os
from PIL import Image
from pytesseract import Output
from matplotlib.patches import Rectangle
import numpy as np

"""
Page segmentation modes:

0. Orientation and script detection (OSD) only.

1. Automatic page segmentation with OSD.

2. Automatic page segmentation, but no OSD, or OCR. (not implemented)

3. Fully automatic page segmentation, but no OSD. (Default)

4. Assume a single column of text of variable sizes.

5. Assume a single uniform block of vertically aligned text.

6. Assume a single uniform block of text.

7. Treat the image as a single text line.

8. Treat the image as a single word.

9. Treat the image as a single word in a circle.

10. Treat the image as a single character.

11. Sparse text. Find as much text as possible in no particular order.

12. Sparse text with OSD.

13. Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.

OCR Engine modes:

Legacy engine only.
Neural nets LSTM engine only.
Legacy + LSTM engines.
Default, based on what is available.
"""

pytesseract.pytesseract.tesseract_cmd = r'C:/Users/yewji/FYP_20297501/server/text_detection/Tesseract-OCR/tesseract.exe'

image_path = ("C:/Users/yewji/FYP_20297501/server/text_detection/images/invoice-sample.jpg")

# Read the image using cv2
original_image = cv2.imread(image_path)

# Check if the image is read successfully
if original_image is not None:

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Grey scale
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    # Image resizing
    resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Denoise image
    gaussian_blur_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # Get word-level bounding box coordinates and text information
    d = pytesseract.image_to_data(gaussian_blur_image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
    n_boxes = len(d['level'])

    # Define the rotation angle increment
    rotation_angle_increment = 30

    rotated_image = gaussian_blur_image

    # Iterate through rotation angles from 0 to 360 in increments of 30 degrees
    for rotation_angle in range(0, 360, rotation_angle_increment):
        if rotation_angle > 0:
            # Create rotation matrix
            matrix = cv2.getRotationMatrix2D((rotated_image.shape[1] // 2, rotated_image.shape[0] // 2), rotation_angle, 1.0)
            # Apply rotation to the image
            rotated_image = cv2.warpAffine(gaussian_blur_image, matrix, (rotated_image.shape[1], rotated_image.shape[0]))

        # Draw bounding boxes and text
        d = pytesseract.image_to_data(rotated_image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
        n_boxes = len(d['level'])

        for i in range(n_boxes):
            if int(d['conf'][i]) > 0:
                x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                if rotation_angle == 0:
                    rotated_x, rotated_y, rotated_w, rotated_h = x, y, w, h
                else:
                    # Rotate the bounding box coordinates
                    rotated_x = int(x * np.cos(np.radians(rotation_angle)) - y * np.sin(np.radians(rotation_angle)))
                    rotated_y = int(x * np.sin(np.radians(rotation_angle)) + y * np.cos(np.radians(rotation_angle)))
                    rotated_w = w
                    rotated_h = h
                # Draw the rotated bounding box on the rotated image
                cv2.rectangle(rotated_image, (rotated_x, rotated_y), (rotated_x + rotated_w, rotated_y + rotated_h), (0, 255, 0), 2)
                # Draw the detected text next to the bounding box
                cv2.putText(rotated_image, d['text'][i], (rotated_x, rotated_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Print the detected text for the current bounding box
                print(f"Detected text for Box {i+1} (rotated {rotation_angle} degrees): {d['text'][i]}")

        # Show the rotated and annotated image
        #plt.imshow(rotated_image)
        #plt.axis('off')
        #plt.title(f'Rotated Detection ({rotation_angle} degrees)')
        #plt.show()


else:
    print("Error reading the image.")
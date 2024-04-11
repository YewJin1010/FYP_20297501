from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trOCR model
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')

# Load Tesseract model
pytesseract.pytesseract.tesseract_cmd = r'text_detection/Tesseract-OCR/tesseract.exe'

# Load image
image_path = 'text_detection/sample_images/yeast.jpg'
image = cv2.imread(image_path)
original_image = image.copy()  # Make a copy for drawing bounding boxes later

# Text detection using trOCR model
trocr_image = Image.open(image_path).convert("RGB")
pixel_values = processor(trocr_image, return_tensors="pt").pixel_values
generate_ids = trocr_model.generate(pixel_values)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print("Text detected using trOCR model:", generated_text)

# Text detection using Tesseract model
text_detected = []
tesseract_text = pytesseract.image_to_data(image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
for i in range(len(tesseract_text['text'])):
    if int(tesseract_text['conf'][i]) > 50:  # Only consider confident detections
        text_detected.append(tesseract_text['text'][i])
print("Text detected using Tesseract model:", text_detected)

# Draw bounding boxes around the text detected with Tesseract model
for i in range(len(tesseract_text['text'])):
    if int(tesseract_text['conf'][i]) > 50:  # Only consider confident detections
        left = tesseract_text['left'][i]
        top = tesseract_text['top'][i]
        width = tesseract_text['width'][i]
        height = tesseract_text['height'][i]
        text = tesseract_text['text'][i]
        cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255), 2)  # Red color
        cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # Red color

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Text Detection using Tesseract")
plt.axis('off')
plt.show()

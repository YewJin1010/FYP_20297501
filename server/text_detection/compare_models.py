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
pytesseract.pytesseract.tesseract_cmd = r'server/text_detection/Tesseract-OCR/tesseract.exe'

# Load image
image_path = 'server/text_detection/sample_images/yeast.jpg'
image = cv2.imread(image_path)

# Text detection using trOCR model
trocr_image = Image.open(image_path).convert("RGB")
pixel_values = processor(trocr_image, return_tensors="pt").pixel_values
generate_ids = trocr_model.generate(pixel_values)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print("Text detected using trOCR model:", generated_text)

# Text detection using Tesseract model
text_detected = []
tesseract_text = pytesseract.image_to_data(image, lang='eng', config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
text_detected.append(tesseract_text['text'])
print("Text detected using Tesseract model:", text_detected)

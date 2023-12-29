from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTModel,
    VisionEncoderDecoderModel,
)
import requests
from PIL import Image

# Save the processor and model configurations
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor.save_pretrained("test")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
model.config.save_pretrained("test")
model.save_pretrained("test")

# Load image from the IAM dataset
#url = "water_bottle.jpg"
url = "text_image.png"
image = Image.open(url).convert("RGB")

# Load processor and model from the saved configurations
processor = TrOCRProcessor.from_pretrained("test")
model = VisionEncoderDecoderModel.from_pretrained("test")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)

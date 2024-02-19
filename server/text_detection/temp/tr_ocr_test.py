from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

def save_model_and_processor(model_name, output_dir):
    processor = TrOCRProcessor.from_pretrained(model_name)
    processor.save_pretrained(output_dir)
    
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

def generate_text_from_image(image_path, model_dir):
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text

if __name__ == "__main__":
    model_name = "microsoft/trocr-base-handwritten"
    output_directory = "test"
    save_model_and_processor(model_name, output_directory)

    image_path = "text_image.png"
    generated_text = generate_text_from_image(image_path, output_directory)
    print(generated_text)

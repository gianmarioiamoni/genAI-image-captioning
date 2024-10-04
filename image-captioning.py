# install the transformers library
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", clean_up_tokenization_spaces=False)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the image
image = Image.open("image.jpg")

# Preprocess the image
inputs = processor(image, return_tensors="pt")

# Generate the caption
outputs = model.generate(**inputs, max_new_tokens=100)

# Decode the caption
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated caption:", caption)

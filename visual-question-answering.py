import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = BlipForConditionalGeneration.from_pretrained("openai/clip-vit-base-patch32")    

# Load the image
img_url = "https://images.unsplash.com/photo-1575936123452-b67c3203c357?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8Mnx8aW1hZ2V8ZW58MHx8MHx8&w=1000&q=80"
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(text, images=raw_image, return_tensors="pt", padding=True)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, text, return_tensors="pt", padding=True)

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

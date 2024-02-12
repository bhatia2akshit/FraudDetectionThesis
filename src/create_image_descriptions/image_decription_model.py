import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import VQGPT2LMHeadModel, VQGPT2Tokenizer

# Load pre-trained VQGPT-2 model and tokenizer
tokenizer = VQGPT2Tokenizer.from_pretrained("EleutherAI/vqgan-gpt2-imagenet")
model = VQGPT2LMHeadModel.from_pretrained("EleutherAI/vqgan-gpt2-imagenet")

# Define preprocessing function for images
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to required size
        transforms.ToTensor()           # Convert image to PyTorch tensor
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Generate descriptions for images
def generate_descriptions(image_paths):
    descriptions = []
    for image_path in image_paths:
        # Preprocess image
        image = preprocess_image(image_path)
        # Generate description
        input_text = tokenizer.decode(tokenizer.bos_token_id)
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids=input_ids.to(model.device),
                                pixel_values=image.to(model.device))
        description = tokenizer.decode(output[0], skip_special_tokens=True)
        descriptions.append(description)
    return descriptions

# Example usage


# Example usage

main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/'
image_folder = main_folder + 'data/real/lsun/bedroom/'
image_paths = [image_folder+"1.jpg", image_folder+"2.jpg", image_folder+"3.jpg"]
captions = generate_descriptions(image_paths)
for image_path, caption in zip(image_paths, captions):
    print(f"Image: {image_path}, Caption: {caption}")



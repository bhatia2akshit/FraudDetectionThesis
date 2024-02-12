import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "HuggingFaceM4/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/'
image_folder = main_folder + 'data/real/lsun/bedroom/'
image_paths = [image_folder + "1.jpg", image_folder + "2.jpg", image_folder + "3.jpg"]
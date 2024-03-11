import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import Dataset, load_dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPModel

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

dataset = load_dataset("imagefolder", data_dir='./Dataset1')
dataloaded = DataLoader(dataset, batch_size=64, shuffle=True)

def get_image_embeddings(loader):
    # Create a DataLoader for the images

    embeddings = []
    # Iterate over batches of images
    for batch in loader:
        # Load and preprocess the images using the CLIP processor
        images = processor(images=batch, return_tensors="pt", padding=True).to(device)

        # Pass the preprocessed images through the CLIP model to obtain the image embeddings
        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=images.pixel_values)

        # Append the image embeddings to the list
        embeddings.append(image_features)

    # Concatenate the embeddings from all batches
    embeddings = torch.cat(embeddings)

    return embeddings

image_embeddings = get_image_embeddings(dataloaded )
# Define a simple linear classifier
class ImageClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ImageClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


import os
import pandas as pd
from torchvision.io import read_image


# Define preprocessing steps for images
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# Function to classify an image
def classify_image(image_tensor, classifier):
    image_input = image_tensor.to(device)

    # Get the image embedding from CLIP
    with torch.no_grad():
        image_embedding, _ = model.encode_image(image_input)

    # Use the classifier to predict if the image is real or fake
    prediction = classifier(image_embedding)

    # Convert prediction to label
    label = "real" if prediction.item() >= 0.5 else "fake"
    confidence = prediction.item()

    return label, confidence


# Example usage:
if __name__ == "__main__":
    # Load the pre-trained classifier
    classifier = ImageClassifier(768)  # CLIP's embedding size is 768
    # classifier.load_state_dict(torch.load("classifier.pth"))  # Load your trained classifier

    train_folder_path = '/data/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/real/coco/coco'
    # Example usage with data loader
    train_dataset = ImageFolder("train_folder_path", transform=preprocess)
    test_dataset = ImageFolder("test_folder_path", transform=preprocess)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Training loop
    for images, labels in train_loader:
        embeddings = []
        for image in images:
            image_embedding, _ = model.encode_image(image.unsqueeze(0).to(device))
            embeddings.append(image_embedding)
        embeddings = torch.stack(embeddings)
        predictions = classifier(embeddings)
        # Further steps for training...

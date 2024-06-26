import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, DDIMScheduler
import torch.nn as nn
import numpy as np
from src.Dataset1.neural_network import basic_neural_network

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained stable diffusion model and DDIM scheduler
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                        torch_dtype=torch.float16).to('cuda')
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Move pipeline to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

tensor2image = transforms.ToPILImage()

def reconstruct_loader(data_loader):
    dire_scores = []
    labels_list = []
    for images, labels in data_loader:
        prompts = ["draw a high quality image"] * len(images)
        images = [tensor2image(image) for image in images]
        reconstructions = pipe(prompt=prompts, image=images, strength=0.1, guidance=1).images

        for original, reconstructed, label in zip(images, reconstructions, labels):
            reconst = transform(reconstructed)
            original = transform(original)
            dire_score = torch.abs(original - reconst)
            dire_scores.append(dire_score)
            labels_list.append(label)

    dire_scores = np.array(dire_scores)
    labels = np.array(labels_list)

    return dire_scores, labels


def main():
    # Define transforms for the images

    main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/Dataset1'
    train_folder = os.path.join(main_folder, 'train')
    test_folder = os.path.join(main_folder, 'test')
    valid_folder = os.path.join(main_folder, 'validation')

    # Combine datasets and create DataLoader
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dataset = datasets.ImageFolder(valid_folder, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_dataset = datasets.ImageFolder(test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    train_dire_scores, train_labels = reconstruct_loader(train_loader)
    train_dataset = torch.utils.data.TensorDataset(train_dire_scores, train_labels)
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    val_dire_scores, val_labels = reconstruct_loader(val_loader)
    val_dataset = torch.utils.data.TensorDataset(val_dire_scores, val_labels)
    validloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

    test_dire_scores, test_labels = reconstruct_loader(test_loader)
    test_dataset = torch.utils.data.TensorDataset(test_dire_scores, test_labels)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)

    # Modify the classifier part of ResNet50
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Number of classes in your dataset

    model = model.to(device)
    basic_neural_network(model, trainloader, validloader, testloader, model_name='dire_report',n_epochs = 20)

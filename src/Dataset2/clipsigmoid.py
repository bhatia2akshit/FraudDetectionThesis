import torch
import torch.nn as nn
from transformers import CLIPImageProcessor, CLIPModel
from datasets import Dataset, load_dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPModel
import os
import torchvision
from torchvision import datasets, transforms, models
from src.Dataset1.neural_network import basic_neural_network

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPModelClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(CLIPModelClassifier, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

        self.fc = nn.Linear(768, 1)

    def forward(self, x):
        features = self.model.get_image_features(pixel_values=x)
        logits = self.fc(features)
        return logits

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"


    main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/Dataset1'
    train_folder = os.path.join(main_folder,'train')
    test_folder = os.path.join(main_folder, 'test')
    valid_folder = os.path.join(main_folder, 'validation')


    #transformations
    transformations = transforms.Compose([transforms.Resize((224,224)),
                                           transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225],),
                                           ])


    #datasets
    train_data = datasets.ImageFolder(train_folder, transform=transformations)
    valid_data = datasets.ImageFolder(valid_folder, transform=transformations)
    test_data = datasets.ImageFolder(test_folder, transform=transformations)

    #dataloader
    trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle = True, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)



    model = CLIPModelClassifier().to(device)

    # freeze all paramas of this model, because we are using a pretrained CLIP model
    for params in model.parameters():
        params.requires_grad_ = False

    basic_neural_network(model, trainloader, validloader, testloader,
                                                         model_name='clipSigmoid')


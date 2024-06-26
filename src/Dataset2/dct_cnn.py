import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from src.Dataset1.neural_network import basic_neural_network
import torch_dct as dct

device = "cuda" if torch.cuda.is_available() else "cpu"

class DCTTransform:
    def __call__(self, image):
        image_array = np.array(image)
        dct_image = dct.dct_2d(torch.tensor(image_array, dtype=torch.float32))
        return dct_image

class DCT_CNN(nn.Module):
    def __init__(self):
        super(DCT_CNN, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * 14 * 14, 1)  # Corrected input size based on the calculations

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    main_folder = '/upb/users/b/bakshit/profiles/unix/cs/FraudDetectionThesis/data/Dataset1'
    train_folder = os.path.join(main_folder,'train')
    test_folder = os.path.join(main_folder, 'test')
    valid_folder = os.path.join(main_folder, 'validation')

    # Transformations based on dct application
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        DCTTransform()
    ])

    #datasets
    train_data = datasets.ImageFolder(train_folder, transform=transformations)
    valid_data = datasets.ImageFolder(valid_folder, transform=transformations)
    test_data = datasets.ImageFolder(test_folder, transform=transformations)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
    validloader = torch.utils.data.DataLoader(valid_data, shuffle = True, batch_size=16)
    testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)

    model = DCT_CNN()
    for params in model.parameters():
      params.requires_grad_ = False

    model = model.to(device)

    basic_neural_network(model, trainloader, validloader, testloader, model_name='dct_cnn',n_epochs = 20)



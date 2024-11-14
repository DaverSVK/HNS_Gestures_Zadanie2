import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import models, datasets
import torchvision.transforms as transforms
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST 
from torch.utils.tensorboard import SummaryWriter
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data
import torch.optim as optim

class CustomCNN0(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomCNN0, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(16 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomCNN1(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomCNN1, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(16 * 11 * 11, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomCNN2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomCNN2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(16 * 10 * 10, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CustomCNN4(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()  # input 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # 24x24 1 input (obrazok) 10 neuronov
            nn.ReLU(),
            nn.BatchNorm2d(10), # 10 neuronov
            nn.MaxPool2d(kernel_size=2, stride=2), #12x12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), #10 vstupov (z predosleho layeru neurony) #20 neuronov 8x8 size
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2), #4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(320, 100), # 20x4x4 4x4 v kazdom neurone, je ich 20 tych neuronov duh
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),  # Output a single logit for binary classification
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

class CustomMLP(nn.Module):
    def __init__(self):
        super(CustomMLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(28*28, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),  # Output a single logit for binary classification
        )

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

class CustomCNN4(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()  # input 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5), # 24x24 1 input (obrazok) 10 neuronov
            nn.ReLU(),
            nn.BatchNorm2d(10), # 10 neuronov
            nn.MaxPool2d(kernel_size=2, stride=2), #12x12
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5), #10 vstupov (z predosleho layeru neurony) #20 neuronov 8x8 size
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=2, stride=2), #4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(320, 100), # 20x4x4 4x4 v kazdom neurone, je ich 20 tych neuronov duh
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1),  # Output a single logit for binary classification
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.shape[0], -1)
        out = self.fc_layers(out)
        return out

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from config import NUM_CLASSES

class DogEmotionResNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        return self.model(x)
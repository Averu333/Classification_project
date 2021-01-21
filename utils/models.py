import torch
import torchvision
from torchvision import models

def get_classifier_model(model_name, num_classes,  use_pretrained=True):
    if model_name == 'resnet':
        model = models.resnet50(pretrained=use_pretrained)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    
    return model
import torch
import torchvision
from torchvision import models

def get_classifier_model(model_name, num_classes, use_pretrained=True):
    '''
    A function to construct network model.
    Args:
        model_name (string): name of the model to be used, can be one of following 'resnet'
        num_classes (int): Number of output classes.
        use_pretrained (bool): If True uses pretrained version of the
                               network if a pretrained version is available.
    Returns:
        model (torch model dict): Constructed model.
    '''
    
    if model_name == 'resnet':
        model = models.resnet50(pretrained=use_pretrained)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)
    
    return model
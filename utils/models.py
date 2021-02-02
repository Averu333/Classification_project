import torch
import torchvision
from torchvision import models
from torch import nn

def turn_grad_off(model, activate=True):
    if activate:
        for param in model.parameters():
            param.requires_grad = False

def get_classifier_model(model_name, num_classes, use_pretrained=True, train_last_layer_only=False):
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
        turn_grad_off(model, train_last_layer_only)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, num_classes)

    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=use_pretrained)
        turn_grad_off(model, train_last_layer_only)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    elif model_name == 'vgg':
        model = models.vgg11_bn(pretrained=use_pretrained)
        turn_grad_off(model, train_last_layer_only)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    elif model_name == 'squeezenet':
        model = models.squeezenet1_0(pretrained=use_pretrained)
        turn_grad_off(model, train_last_layer_only)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes

    elif model_name == 'densenet':
        model = models.densenet121(pretrained=use_pretrained)
        turn_grad_off(model, train_last_layer_only)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)

    elif model_name == 'mymodel':
        model = mymodel(num_classes)

    else:
        print("Invalid model name. Exiting.")
        exit()
    
    return model

class mymodel(nn.Module):
    '''
    My own network model. Put together a network for demonstration.
    If a proper network is needed strides should be shorter, more layers etc.
    '''
    def __init__(self, num_classes):
        super(mymodel, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        #224 -> 224
        self.conv_layer1 = self.make_conv_layer(in_channels=3,
                                           out_channels=64,
                                           kernel_size=3,
                                           stride=1,
                                           padding=1,
                                           maxpool=2)
        #112 -> 26
        self.conv_layer2 = self.make_conv_layer(in_channels=64,
                                           out_channels=128,
                                           kernel_size=4,
                                           stride=1,
                                           padding=0,
                                           maxpool=2)
        #26 -> 12
        self.conv_layer3 = self.make_conv_layer(in_channels=128,
                                           out_channels=128,
                                           kernel_size=2,
                                           stride=1,
                                           padding=0,
                                           maxpool=2)
        self.fc1 = nn.Linear(12*12*128, 2048)
        self.fc2 = nn.Linear(2048, 245)
        self.fc3 = nn.Linear(245, num_classes)
    
    def make_conv_layer(self, in_channels, out_channels, kernel_size, stride, padding, maxpool):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        conv_layer = nn.Sequential(conv2d, nn.BatchNorm2d(out_channels), self.relu, nn.MaxPool2d(kernel_size=maxpool))
        return conv_layer
    
    def forward(self, x):
        #Conv layers
        y = self.conv_layer1(x)
        y = self.maxpool(y)
        y = self.conv_layer2(y)
        y = self.conv_layer3(y)

        #Linear layers
        y = y.view(y.size(0), -1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.relu(y)
        y = self.fc3(y)        
        return y
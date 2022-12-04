from config import NUM_CLASSES, ALEXNET_HIDDEN_UNITS, RESNET_USE_CUDA
from torch import nn
from torchvision import models
from collections import OrderedDict
from collections import namedtuple

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, device, feature_extract=True, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None

    if model_name == "resnet":
        """ Resnet50
        """
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        model.fc = nn.Sequential(
               nn.Linear(2048, 128),
               nn.ReLU(inplace=True),
               nn.Linear(128, NUM_CLASSES))
        model.to(device)
        
    elif model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model, feature_extract)
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(9216, ALEXNET_HIDDEN_UNITS)), # this should be a Linear layer; how to connect with the pretrained model?
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.5)),
            ('fc2', nn.Linear(ALEXNET_HIDDEN_UNITS, NUM_CLASSES)), # what is the output of this layer should be
            ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        model.to(device)
        
    else:
        print("Invalid model name, exiting...")
        exit()

    return model
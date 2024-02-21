import torch
from torchvision import models

def set_model(name, path):
    model = get_model(name)
    # model.load_state_dict(torch.load(path))
    return model

def get_model(name):
    n_way = 10
    name = name.lower()
    if name == 'resnet18':
        model = models.resnet18(pretrained='IMAGENET1K_V2')
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet34":
        model = models.resnet34()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet50":
        model = models.resnet50()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet101":
        model = models.resnet101()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="resnet152":
        model = models.resnet152()
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif str=="alexnet":
        model = models.alexnet()
        num_ftrs = model.classifier._modules["6"].in_features
        model.classifier._modules["6"] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.features._modules["0"] = torch.nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False)
        return model
    elif str=="densenet121":
        model = models.densenet121()
        num_ftrs = model.classifier.in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        return model
    elif str=="mobilenet_v2":
        model = models.mobilenet_v2()
        num_ftrs = model.classifier._modules["1"].in_features
        model.classifier._modules["1"] = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(num_ftrs, n_way)
        )
        model.features._modules["0"]._modules["0"] = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    else:
        print("wrong model")
        print("possible models : resnet18, resnet34, resnet50, resnet101, resnet152, alexnet, densenet121, mobilenet_v2")
        exit()
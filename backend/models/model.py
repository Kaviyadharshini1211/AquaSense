# models/model.py
import torch.nn as nn
from torchvision import models

def get_model(num_classes=3, in_channels=6):
    # Load pretrained ResNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Replace the first conv to take `in_channels` instead of 3
    old = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,                # ← now 6
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None
    )

    # (Re)initialize its weights
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

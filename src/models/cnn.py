"""Torchvision ResNet18 baseline factory."""

from __future__ import annotations


def make_resnet18_binary(pretrained: bool = True):
    import torch
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

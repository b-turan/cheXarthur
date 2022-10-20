import timm
import torch
from torch import nn

def init_model(model_name: str) -> torch.nn.Module:
    """Initialize Model."""
    m = timm.create_model(model_name, pretrained=False, num_classes=14)
    return m


class DummyClassifier(nn.Module):
    """Dummy classifier to start experiments."""

    def __init__(self):
        super().__init__()
        self.fw_layers = nn.Sequential(nn.Linear(320 * 320, 50), nn.ReLU(), nn.Linear(50, 14))

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fw_layers(x)
        return x

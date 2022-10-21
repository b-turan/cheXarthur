import torch
from torch import nn
from torchvision import models

# TODO: Write dict that maps model name to corresponding model

def initialize_model(
    model_name: str, n_channels: int, pre_trained: bool
) -> torch.Tensor:
    """Initializes Model.

    Args:
        model_name (str): Model name.
        n_channels (int): Number of input channels.
        pre_trained (bool): Pretrained or from scratch initialization.

    Returns:
        torch.tensor: Artificial Neural Net.
    """
    assert model_name == "resnet18", f"Only supports ``resnet18`` at the moment, got {model_name}."
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pre_trained else None
    model = models.resnet18(weights)
    model.fc = nn.Linear(512, 14)  # Adapt output layer
    if n_channels == 1:
        # Adapt resnet18 to one-channel input
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return model

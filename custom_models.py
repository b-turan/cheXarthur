import torch
from torch import nn
from torchvision import models

# TODO: Write dict that maps model name to corresponding model


def initialize_model(model_name: str, n_channels: int, pre_trained: bool) -> torch.Tensor:
    """Initializes Model.

    Args:
        model_name (str): Model name.
        n_channels (int): Number of input channels.
        pre_trained (bool): Pretrained or from scratch initialization.

    Returns:
        torch.tensor: Artificial Neural Net.
    """
    # fmt: off
    assert model_name == "densenet121", f"Only supports ``densenet121`` at the moment, got {model_name}."
    # fmt: on
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pre_trained else None
    model = models.densenet121(weights)

    # Adapt input layer
    model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Adapt output layer
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 14)

    return model

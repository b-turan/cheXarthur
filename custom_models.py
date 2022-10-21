import torch
from torch import nn
from torchvision import models

# TODO: Write dict that maps model name to corresponding model


def initialize_model(
    model_name: str, n_classes: int, n_channels: int, pre_trained: bool
) -> nn.Module:
    """Initializes Model.

    Args:
        model_name (str): Model name.
        n_classes (int): Number of classes.
        n_channels (int): Number of input channels.
        pre_trained (bool): Pretrained or from scratch initialization.

    Returns:
        nn.Module: Artificial Neural Net.
    """
    # fmt: off
    assert model_name in ("densenet121", "resnet18"), f"Only supports ``densenet121`` or ``resnet18`` for now, got ``{model_name}``."
    # fmt: on

    model_zoo = {
        "densenet121": DenseNet121,
        "resnet18": ResNet18,
    }
    ModelConstructor = model_zoo[model_name]
    model = ModelConstructor(n_classes=n_classes, n_channels=n_channels, pre_trained=pre_trained)

    return model


class DenseNet121(nn.Module):
    """Modified DenseNet121 model.

    Standard DenseNet121 except the input/output layer are adapted to the cheXpert dataset.
    """

    def __init__(self, n_classes: int, n_channels: int, pre_trained: bool):
        """Initializes DenseNet121. Modifies input and output layer.

        Args:
            n_classes (int): Number of classes.
            n_channels (int): Number of channels.
            pre_trained (bool): If true, loads pre-trained weights.
        """
        super().__init__()
        weights = "DenseNet121_Weights.DEFAULT" if pre_trained else None
        self.densenet121 = models.densenet121(weights=weights)

        self.densenet121.features.conv0 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Adapt output layer
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ResNet18(nn.Module):
    """Modified ResNet18 model.

    Standard ResNet18 except the input/output layer are adapted to the cheXpert dataset.
    """

    def __init__(self, n_classes: int, n_channels: int, pre_trained: bool):
        """Initializes ResNet18. Modifies input and output layer.

        Args:
            n_classes (int): Number of classes.
            n_channels (int): Number of channels.
            pre_trained (bool): If true, loads pre-trained weights.
        """
        super().__init__()
        weights = "ResNet18_Weights.DEFAULT" if pre_trained else None
        self.resnet18 = models.resnet18(weights=weights)

        # Adapt input layer
        self.resnet18.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Adapt output layer
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        x = self.resnet18(x)
        return x


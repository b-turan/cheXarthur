import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from chexpert_dataset import CheXpertDataset
from models import DummyClassifier, init_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device:", device)

# keep in mind for loss function:
# https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804
# https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/4


# warning: The output image might be different depending on its type: when downsampling,
# the interpolation of PIL images and tensors is slightly different, because PIL applies antialiasing.
# This may lead to significant differences in the performance of a network.
# Therefore, it is preferable to train and serve a model with the same input types.
# See also below the antialias parameter, which can help making the output of PIL images and tensors closer.


def main():
    train_data = CheXpertDataset(
        root_dir=".data/",
        split="train",
        transform=transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()]),
    )
    valid_data = CheXpertDataset(
        root_dir=".data/",
        split="valid",
        transform=transforms.Compose([transforms.Resize([320, 320]), transforms.ToTensor()]),
        target_transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, num_workers=4)

    model = init_model("inception_v4")

    for x, y in tqdm(train_loader):
        print("hi function")

if __name__ == "__main__":
    main()

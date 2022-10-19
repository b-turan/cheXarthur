import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from chexpert_dataset import CheXpertDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device:", device)


def main():
    train_data = CheXpertDataset(root_dir=".data/", split="train")
    valid_data = CheXpertDataset(root_dir=".data/", split="valid")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=16, shuffle=False, num_workers=4)


if __name__ == "__main__":
    main()

from __future__ import division, print_function

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import ToTensor

# TODO: Either load entire dataset into RAM or ChunkDataset for speedups
# see https://github.com/pytorch/pytorch/pull/21232


class CheXpertDataset(Dataset):
    """
    CheXpert is a large dataset of chest X-rays and competition for automated chest
    x-ray interpretation, which features uncertainty labels and radiologist-labeled
    reference standard evaluation sets.

    source: https://stanfordmlgroup.github.io/competitions/chexpert/
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            root_dir (str): Relative path to directory, where the dataset is located.
            split (string): The dataset split, supports ``train``, or ``valid``.
            transform (Optional[Callable], optional): Transforms images. Defaults to None.
            target_transform (Optional[Callable], optional):  Transforms targets (e.g., ignore strategies). Defaults to None.
        """
        # fmt: off
        assert split=="train" or split=="valid", f"Expected support for split is ``train`` or ``valid``, got ``{split}``"
        assert root_dir[-1] == "/", f"Expected ``root_dir`` to have ``/`` at the end of the string, got ``{root_dir}``"
        # fmt: on

        self.data = pd.read_csv(root_dir + f"CheXpert-v1.0-small/{split}.csv")
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        cxr_img = read_image(img_name)
        label = self.data.iloc[idx, 5:]
        label = np.array([label])

        if self.transform:
            cxr_img = self.transform(cxr_img)
        if self.target_transform:
            label = self.target_transform(label)

        return cxr_img, label

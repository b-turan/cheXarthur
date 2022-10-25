from __future__ import division, print_function

import os
from typing import Any, Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

# TODO: Load entire dataset into RAM or ChunkDataset for speedups
# see https://github.com/pytorch/pytorch/pull/21232


class CheXpertDataset(Dataset):
    """Dataset of chest-xray images.

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
        ignore_index: Optional[int] = -100,
        policy: Optional[str] = "ignore",
        n_channels: Optional[int] = 1,
    ) -> None:
        """
        Args:
            root_dir (str): Relative path to directory, where the dataset is located.
            split (string): The dataset split, supports ``train``, or ``valid``.
            transform (Callable, optional): Transforms images. Defaults to None.
            ignore_index (int, optional): Replaces all NaN's by ignore_index, which are ignored in later loss calculations.
            policy (str, optional): Determines how to deal with ``uncertainty label``,
                supports ``ignore``, ``ones`` and ``zeros``.
            n_channels (int, optional): Number of color channels, supports 1 and 3.
                If n_channels=3, R=G=B=Grayscale, which is useful since it allows loading pretrained models.
        """
        # fmt: off
        assert split in ("train", "valid"), f"Expected support for split is ``train`` or ``valid``, got ``{split}``"
        assert root_dir[-1] == "/", f"Expected ``root_dir`` to have ``/`` at the end of the string, got ``{root_dir}``"
        assert policy in ("ignore", "ones", "zeros"), f"Expected support for ``policy`` is ``ignore``, ``ones`` or ``zeros``, got ``{policy}``"
        assert n_channels in (1, 3), f"Expected support for ``n_channels`` is 1 or 3, got ``{n_channels}``"
        # fmt: on

        self.ignore_index = ignore_index
        self.policy = policy
        self.n_channels = n_channels

        # Read data and replace ``NaN`` with ignore_index
        self.df = pd.read_csv(root_dir + f"CheXpert-v1.0-small/{split}.csv")
        self.df.fillna(value=self.ignore_index, inplace=True)

        # self.df_frontal = self.df[self.df["Path"].str.contains("frontal")].copy()
        # self.df_lateral = self.df[self.df["Path"].str.contains("lateral")].copy()

        if policy == "ignore":
            self.df.replace(to_replace=-1, value=-100, inplace=True)
        elif policy == "ones":
            self.df.replace(to_replace=-1, value=1, inplace=True)
        elif policy == "zeros":
            self.df.replace(to_replace=-1, value=0, inplace=True)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])

        if self.n_channels == 1:
            cxr_img = Image.open(img_name)
        elif self.n_channels == 3:
            cxr_img = Image.open(img_name).convert("RGB")

        label = self.df.iloc[idx, 5:]
        label = torch.tensor([label], dtype=torch.float16).squeeze()

        if self.transform:
            cxr_img = self.transform(cxr_img)

        return cxr_img, label

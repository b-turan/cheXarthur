import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.functional.classification import multilabel_precision
from torchvision import models, transforms
from tqdm import tqdm

from chexpert_dataset import CheXpertDataset
from models import DummyClassifier

# keep in mind for loss function:
# https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804
# https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/4


# NOTE: The output image might be different depending on its type: when downsampling,
# the interpolation of PIL images and tensors is slightly different, because PIL applies antialiasing.
# This may lead to significant differences in the performance of a network.
# Therefore, it is preferable to train and serve a model with the same input types.
# See also below the antialias parameter, which can help making the output of PIL images and tensors closer.


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on Device:", device)

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
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False, num_workers=4)

    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 14)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Metrics
    ign_idx = -1  # ``uncertain`` label
    accuracy_metric = BinaryAccuracy(ignore_index=ign_idx).to(device)
    f1_metric = MultilabelF1Score(num_labels=14, ignore_index=ign_idx, average=None).to(device)
    precision_metric = MultilabelPrecision(num_labels=14, ignore_index=ign_idx, average=None).to(
        device
    )
    recall_metric = MultilabelRecall(num_labels=14, ignore_index=ign_idx, average=None).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-10)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: ", n_params)

    model.train()
    epoch_loss = 0
    count_samples = torch.zeros(14).to(device)
    for x, y in tqdm(train_loader):
        x = x.to(device)
        y = y.to(device)

        # Inference
        logits = model(x)
        loss = criterion(logits, y)

        # Mask to ignore ``NaN`` and ``uncertainty`` labels
        mask = y != ign_idx
        loss = torch.masked_select(loss, mask).mean()

        epoch_loss += loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        # Metrics
        preds = torch.sigmoid(logits)
        mean_acc = accuracy_metric(preds, y)
        f1_score = f1_metric(preds, y)
        precision_score = precision_metric(preds, y)
        recall_score = recall_metric(preds, y)

        count_samples += mask.sum(dim=0)


if __name__ == "__main__":
    main()

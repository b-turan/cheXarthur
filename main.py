import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchvision import models, transforms
from tqdm import tqdm

from chexpert_dataset import CheXpertDataset
from custom_models import initialize_model

# keep in mind for loss function:
# https://discuss.pytorch.org/t/ignore-padding-area-in-loss-computation/95804
# https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/4


# NOTE: The output image might be different depending on its type: when downsampling,
# the interpolation of PIL images and tensors is slightly different, because PIL applies antialiasing.
# This may lead to significant differences in the performance of a network.
# Therefore, it is preferable to train and serve a model with the same input types.
# See also below the antialias parameter, which can help making the output of PIL images and tensors closer.

# TODO: Data augmentation Normalization, RandomResizedCrop
# TODO: weight loss by class ratio?
# TODO: Write class ``MetricsCheXpert``
# TODO: Integrate multi-view classification

N_CLASSES = 14
IGN_IDX = -100
N_CHANNELS = 3
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "No Finding",
    "Enlarged Cardiom",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
BATCH_SIZE = 64


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on Device:", device)

    train_data = CheXpertDataset(
        root_dir=".data/",
        split="train",
        transform=transforms.Compose(
            [
                transforms.Resize([320, 320]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
        ignore_index=IGN_IDX,
        policy="ignore",
        n_channels=N_CHANNELS,
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = initialize_model(
        model_name="resnet18", n_classes=N_CLASSES, n_channels=N_CHANNELS, pre_trained=False
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none").to(device)

    # Metrics
    accuracy_metric = BinaryAccuracy(ignore_index=IGN_IDX).to(device)
    f1_metric = MultilabelF1Score(num_labels=N_CLASSES, ignore_index=IGN_IDX, average=None).to(
        device
    )
    precision_metric = MultilabelPrecision(
        num_labels=N_CLASSES, ignore_index=IGN_IDX, average=None
    ).to(device)
    recall_metric = MultilabelRecall(num_labels=N_CLASSES, ignore_index=IGN_IDX, average=None).to(
        device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: ", n_params)

    model.train()
    running_loss = np.array([])
    mean_acc = 0
    f1_score = torch.zeros(N_CLASSES).to(device)
    precision_score = torch.zeros(N_CLASSES).to(device)
    recall_score = torch.zeros(N_CLASSES).to(device)
    count_samples = torch.zeros(N_CLASSES).to(device)
    for x, y in (pbar := tqdm(train_loader)):
        x = x.to(device)
        y = y.to(device)

        # Inference
        logits = model(x)
        batch_loss = criterion(logits, y)

        # Mask to ignore ``NaN``
        mask = y != IGN_IDX
        batch_loss = torch.masked_select(batch_loss, mask).mean()

        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        # Metrics
        preds = torch.sigmoid(logits)
        mean_acc += accuracy_metric(preds, y)
        f1_score += f1_metric(preds, y)
        precision_score += precision_metric(preds, y)
        recall_score += recall_metric(preds, y)

        # Calculate running loss for display
        running_loss = np.append(running_loss, batch_loss.detach().cpu().numpy())
        avg_loss = np.mean(running_loss[-100:])
        pbar.set_postfix({"Avg. Loss": avg_loss})
    epoch_loss = np.mean(running_loss)
    mean_acc = mean_acc / len(train_loader)
    mean_f1 = [f1 / n for f1, n in zip(f1_score, count_samples)]
    mean_precision = [precision / n for precision, n in zip(precision_score, count_samples)]
    mean_recall = [recall / n for recall, n in zip(recall_score, count_samples)]

    print("Avg Epoch Loss:", epoch_loss)
    print("Accuracy: ", mean_acc)
    print("F1-Scores: ", mean_f1)
    print("Precisions: ", mean_precision)
    print("Recalls: ", mean_recall)


if __name__ == "__main__":
    main()

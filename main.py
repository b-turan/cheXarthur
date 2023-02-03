import argparse

import numpy as np
import sklearn
import sklearn.model_selection
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from utils import MetricsCollector

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="")
parser.add_argument("--batch_size", type=int, default=32, help="")
parser.add_argument("--lr", type=float, default=0.001, help="")

cfg = parser.parse_args()
print(cfg)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device:", device)


# Load the pre-trained ResNet model
model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False).to(device)
# Initialize Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031
chexpert_dataset = xrv.datasets.CheX_Dataset(
    imgpath=".data/CheXpert-v1.0-small/",
    csvpath=".data/CheXpert-v1.0-small/train.csv",
    transform=transforms.Compose(
        [
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224),
        ]
    ),
    unique_patients=False,
)
# Relabel dataset according to the model's pathologies
xrv.datasets.relabel_dataset(model.pathologies, chexpert_dataset)
# Split the dataset into train and test
dataset_split = sklearn.model_selection.GroupShuffleSplit(
    train_size=0.8, test_size=0.2, random_state=cfg.seed
)
# Get the indices of the train and test sets
train_inds, test_inds = next(
    dataset_split.split(X=range(len(chexpert_dataset)), groups=chexpert_dataset.csv.patientid)
)
# Initialize the train and test datasets
train_dataset = xrv.datasets.SubsetDataset(chexpert_dataset, train_inds)
test_dataset = xrv.datasets.SubsetDataset(chexpert_dataset, test_inds)
# Initialize dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)  # type: ignore
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)  # type: ignore

# Training Routine
model.train()
pathologies = [pathology for pathology in model.pathologies if pathology != ""]
criterion = torch.nn.BCEWithLogitsLoss()
# Initialize a list to store the class-wise AUC-ROC scores for the current batch
batch_class_auc_scores = []
num_classes = 11
metric_collector = MetricsCollector("precision", "recall", "f1")
pbar = tqdm(train_loader, desc=f"Train Epoch")
for sample in pbar:
    images = sample["img"].to(device)  # (batch_size, 1, 224, 224)
    labels = sample["lab"].to(device)  # (batch_size, 18)
    # Get dimensions
    batch_size = images.shape[0]
    # Inference model
    outputs = model(images)
    # Remove the labels that are not present in the dataset and reshape
    labels = labels[outputs != 0.5].view(batch_size, num_classes)
    outputs = outputs[outputs != 0.5].view(batch_size, num_classes)
    # Set uncertain labels reperesented by NaN to 0
    labels[torch.isnan(labels)] = 0
    # Compute the loss
    loss = criterion(outputs, labels)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Predictions
    y_preds = torch.sigmoid(outputs).detach().cpu()
    # Threshold the predictions to obtainy y_preds
    y_preds[y_preds >= 0.5] = 1
    y_preds[y_preds < 0.5] = 0
    # Calculate metrics, recall, precision and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels.cpu(), y_preds.cpu(), average=None
    )

    # Update metrics
    metric_collector.add(precision=precision, recall=recall, f1=f1)
    # Update progress bar
    pbar.set_postfix({"loss": loss.item()})
    pbar.update()

    for class_index in range(num_classes):
        # Get the true labels for the current class
        y_true = labels[:, class_index].cpu()
        # Get the predicted probabilities for the current class
        y_prob = y_preds[:, class_index].cpu()
        if len(np.unique(y_true)) <= 1:
            continue
        # Calculate the AUC-ROC score for the current class
        class_auc = roc_auc_score(y_true, y_prob)
        # Add the current class AUC score to the list
        batch_class_auc_scores.append(class_auc)

avg_precision = metric_collector.average("precision")
avg_recall = metric_collector.average("recall")
avg_f1 = metric_collector.average("f1")
print(f"Train Precision: {avg_precision}")
print(f"Train Recall: {avg_recall}")
print(f"Train F1: {avg_f1}")

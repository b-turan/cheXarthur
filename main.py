import argparse

import sklearn
import sklearn.model_selection
import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from tqdm import tqdm

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
criterion = torch.nn.BCEWithLogitsLoss()
pbar = tqdm(train_loader, desc=f"Train Epoch")
for sample in pbar:
    images = sample["img"].to(device)  # (batch_size, 1, 224, 224)
    labels = sample["lab"].to(device)  # (batch_size, 18)
    # Get dimensions
    batch_size = images.shape[0]
    # Inference model
    outputs = model(images)
    # Remove the labels that are not present in the dataset and reshape
    labels = labels[outputs != 0.5].view(batch_size, 11)
    outputs = outputs[outputs != 0.5].view(batch_size, 11)
    # Set uncertain labels reperesented by NaN to 0
    labels[torch.isnan(labels)] = 0
    # Compute the loss
    loss = criterion(outputs, labels)
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # Update progress bar
    pbar.set_postfix({"loss": loss.item()})
    pbar.update()

import torch
import torchvision.transforms as transforms
import torchxrayvision as xrv
from sklearn.metrics import precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on Device:", device)


# Load the pre-trained ResNet model
model = xrv.models.DenseNet(weights="densenet121-res224-chex", apply_sigmoid=False).to(device)

# CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. https://arxiv.org/abs/1901.07031
dataset_chex = xrv.datasets.CheX_Dataset(
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

xrv.datasets.relabel_dataset(model.pathologies, dataset_chex)

test_loader = torch.utils.data.DataLoader(dataset_chex, batch_size=4, shuffle=False)

# Evaluate the model on the test data
with torch.no_grad():
    model.eval()
    # binary cross entropy loss
    criterion = torch.nn.BCEWithLogitsLoss()
    for sample in test_loader:
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
        # Apply sigmoid to the outputs
        predictions = torch.sigmoid(
            outputs
        )  # TODO: Check if sigmoid is already applied in the model
        # Threshold the outputs
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0
        # Compute the metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), predictions.cpu())
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        break

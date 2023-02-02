import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import models


class cheXarthurTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        train_loader,
        val_loader,
        test_loader,
        epochs,
        log_interval,
        log_dir,
        model_dir,
        model_name,
        save_interval,
        save_best_only,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.log_interval = log_interval
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.save_interval = save_interval
        self.save_best_only = save_best_only

        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_val_loss = 1000
        self.best_epoch = 0

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def prepare_dataset(self, dataset):
        pass

    def initialize_model(self, model):
        pass

    def initialize_optimizer(self, optimizer):
        pass

    def regular_training(self, epochs):
        pass

    
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if batch_idx % self.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(self.train_loader.dataset),
                            100.0 * batch_idx / len(self.train_loader),
                            loss.item(),
                        )
                    )
                    self.writer.add_scalar(
                        "train_loss", loss.item(), epoch * len(self.train_loader) + batch_idx
                    )
            train_loss /= len(self.train_loader)
            self.train_losses.append(train_loss)
            print(f"Train set: Average loss: {train_loss:.4f}")

    def test(self):
        pass

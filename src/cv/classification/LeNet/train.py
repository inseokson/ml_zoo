import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torchvision
from model import LeNet
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--random_seed", type=int, default=42)
args = parser.parse_args()

# Set seed for reproducibility
np.random.seed(args.random_seed)
random.seed(args.random_seed)

torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Paths
path = Path(__file__) / ".." / ".." / ".."
path_data = path / "data"
path_model = path / "model"
path_checkpoint = path_model / "LeNet5.pt"

if not path_data.is_dir():
    os.mkdir(path_data)

if not path_model.is_dir():
    os.mkdir(path_model)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root=path_data / "mnist",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

test_dataset = torchvision.datasets.MNIST(
    root=path_data / "mnist",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

# Make dataloader
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Model, Optimizer and Criterion
model = LeNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = torch.nn.CrossEntropyLoss()

# Check existence of checkpoint
if not path_checkpoint.is_file():
    checkpoint = {
        "epoch": 0,
        "best_accuracy": sys.float_info.min,
        "best_loss": sys.float_info.max,
        "loss": [],
        "accuracy": [],
        "accuracy_test": [],
        "n_without_improvement": 0,
        "done": False,
    }

    prev_n_epochs = 0

else:
    checkpoint = torch.load(path_checkpoint)

    if checkpoint.get("done"):
        raise Exception("This model has already been trained.")

    prev_n_epochs = checkpoint.get("epoch")
    model.load_state_dict(checkpoint.get("model_state_dict"))
    optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

# Training
epochs = trange(args.n_epochs, unit="epoch", desc="Training")
for e in epochs:
    model.train()

    loss = 0
    n_epochs = prev_n_epochs + e + 1

    train_loader_loop = tqdm(
        train_loader, unit="batch", desc=f"Training ({n_epochs}th Epoch)"
    )
    for x, y in train_loader_loop:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(x)
        loss_ = criterion(y_pred, y)

        loss += loss_.item() * x.shape[0]

        loss_.backward()
        optimizer.step()

    loss /= len(train_loader.dataset)

    model.eval()
    n_correct = 0
    n_correct_test = 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred_label = torch.max(y_pred, 1)
            n_correct += (y == y_pred_label).sum()
        accuracy = n_correct / len(train_loader.dataset)

        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            _, y_pred_label = torch.max(y_pred, 1)
            n_correct_test += (y == y_pred_label).sum()
        accuracy_test = n_correct_test / len(test_loader.dataset)

    if accuracy > checkpoint.get("best_accuracy"):
        checkpoint["best_epoch"] = n_epochs
        checkpoint["best_accuracy"] = accuracy
        checkpoint["best_accuracy_test"] = accuracy_test
        checkpoint["best_loss"] = loss
        checkpoint["best_model_state_dict"] = model.state_dict()
        checkpoint["n_without_improvement"] = 0

    else:
        checkpoint["n_without_improvement"] += 1

    checkpoint["model_state_dict"] = model.state_dict()
    checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint["epoch"] = n_epochs
    checkpoint["accuracy"].append(accuracy)
    checkpoint["accuracy_test"].append(accuracy_test)
    checkpoint["loss"].append(loss)

    if checkpoint["n_without_improvement"] == 3:
        checkpoint["done"] = True

    torch.save(checkpoint, path_checkpoint)

    print(
        f"""{datetime.now().time()} | Epoch: {n_epochs} Loss: {loss:.4f} Accuracy: {100 * accuracy:.2f}% Accuracy in test: {100 * accuracy_test:.2f}%"""
    )

    if checkpoint.get("done"):
        break

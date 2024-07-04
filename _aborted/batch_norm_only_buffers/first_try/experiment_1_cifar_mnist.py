# %% Init
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms as T
from tqdm import tqdm
import datasets

torch.set_printoptions(sci_mode=False, linewidth=120)


def setup_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


setup_seed(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %% Define datasets and loaders


def get_datasets():
    setup_seed(42)

    image_size = 28
    transform = T.Compose([T.Resize(image_size), T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])])

    cifar10 = datasets.load_dataset("cifar10")
    train_val_split = cifar10["train"].train_test_split(test_size=0.1, stratify_by_column="label")
    cifar10 = datasets.DatasetDict(
        {"train": train_val_split["train"], "val": train_val_split["test"], "test": cifar10["test"]}
    )
    cifar10 = cifar10.rename_column("img", "image")
    cifar10 = cifar10.cast_column("image", datasets.Image(mode="L"))
    cifar10 = cifar10.map(lambda sample: {"pixel_values": transform(sample["image"])})
    cifar10.set_format("pt", columns=["pixel_values"], output_all_columns=True)

    mnist = datasets.load_dataset("mnist")
    train_val_split = mnist["train"].train_test_split(test_size=0.1, stratify_by_column="label")
    mnist = datasets.DatasetDict(
        {"train": train_val_split["train"], "val": train_val_split["test"], "test": mnist["test"]}
    )
    mnist = mnist.map(lambda sample: {"pixel_values": transform(sample["image"])})
    mnist.set_format("pt", columns=["pixel_values"], output_all_columns=True)

    return cifar10, mnist


def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append(example["pixel_values"])
        labels.append(example["label"])

    pixel_values = torch.stack(images)
    labels = torch.tensor(labels)
    return {"pixel_values": pixel_values, "labels": labels}


def get_data_loader(dataset):
    batch_size = 32

    loader = {}
    for split, data in dataset.items():
        loader[split] = torch.utils.data.DataLoader(data, collate_fn=collate_fn, batch_size=batch_size, pin_memory=True)

    return loader


# %% Load datasets

cifar10, mnist = get_datasets()

cifar10_loaders = get_data_loader(cifar10)
mnist_loaders = get_data_loader(mnist)

# %% Train/eval loops


def train_one_epoch(model, criterion, optimizer, train_loader, device):
    model.train()

    train_loss = 0
    train_accuracy = 0

    for batch in tqdm(train_loader):
        inputs = batch["pixel_values"]
        labels = batch["labels"]
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = outputs.argmax(dim=1)
        batch_accuracy = (y_pred_class == labels).sum().item() / len(y_pred_class)
        train_accuracy += batch_accuracy

    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / len(train_loader)

    return train_loss, train_accuracy


def validate(model, criterion, val_loader, device):
    model.eval()

    test_loss = 0.0
    test_accuracy = 0

    with torch.inference_mode():
        for batch in tqdm(val_loader):
            inputs = batch["pixel_values"]
            labels = batch["labels"]
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            y_pred_class = outputs.argmax(dim=1)
            test_accuracy += (y_pred_class == labels).sum().item() / len(y_pred_class)

    test_loss = test_loss / len(val_loader)
    test_accuracy = test_accuracy / len(val_loader)

    return test_loss, test_accuracy


def train_model(model, loaders: dict):
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(model, criterion, optimizer, loaders["train"], device)

        val_loss, val_accuracy = validate(model, criterion, loaders["val"], device)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model


# %% Train the base model (on cifar10)

model_base_cifar = torchvision.models.resnet18(num_classes=10)
# Make the model take images with 1 channel
model_base_cifar.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_base_cifar = train_model(model_base_cifar, cifar10_loaders)
# %% Train the base model (on mnist)

model_base_mnist = torchvision.models.resnet18(num_classes=10)
model_base_mnist.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_base_mnist = train_model(model_base_mnist, mnist_loaders)

# %% Fine-tune the model on new dataset (mnist)
import copy

model_cifar_to_mnist = copy.deepcopy(model_base_cifar)

# Prepare model for fine tuning

for param in model_cifar_to_mnist.parameters():
    param.requires_grad = False

model_cifar_to_mnist.fc = torch.nn.Linear(in_features=512, out_features=10, bias=True)

model_base_mnist = train_model(model_base_mnist, mnist_loaders)

# %%

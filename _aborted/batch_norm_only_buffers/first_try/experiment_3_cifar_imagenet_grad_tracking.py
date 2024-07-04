# %% Init
import copy
import os
import random

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms as T
from tqdm import tqdm

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

# %% Load the dataset

from torchvision.models import ResNet18_Weights, resnet18

transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

cifar10 = datasets.load_dataset("cifar10")
train_val_split = cifar10["train"].train_test_split(test_size=0.1, stratify_by_column="label")
cifar10 = datasets.DatasetDict(
    {"train": train_val_split["train"], "val": train_val_split["test"], "test": cifar10["test"]}
)
cifar10 = cifar10.rename_column("img", "image")
cifar10 = cifar10.map(lambda sample: {"pixel_values": transform(sample["image"])})
cifar10.set_format("pt", columns=["pixel_values"], output_all_columns=True)


# %% Prepare the dataloaders


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


cifar10_loaders = get_data_loader(cifar10)

# %% Train/eval loops

criterion = torch.nn.CrossEntropyLoss()


def train_one_epoch(model, optimizer, train_loader):
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


def validate(model, val_loader):
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 15

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(model, optimizer, loaders["train"])

        val_loss, val_accuracy = validate(model, loaders["val"])

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model


# %% Load the pretrained model

model_base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_base.eval()

# Modify the final layer to match CIFAR-10 classes
model_base.fc = torch.nn.Linear(model_base.fc.in_features, 10)

val_loss, val_accuracy = validate(model_base, cifar10_loaders["val"])
print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# %% Fine-tune the usual way (last layer)
model_cifar_tuned_last_layer = copy.deepcopy(model_base)

# Prepare model for fine tuning
for param in model_cifar_tuned_last_layer.parameters():
    param.requires_grad = False

model_cifar_tuned_last_layer.fc = torch.nn.Linear(model_base.fc.in_features, 10)

model_cifar_tuned_last_layer = train_model(model_cifar_tuned_last_layer, cifar10_loaders)

# %% Fine-tune only batch norm
model_cifar_tuned_normalization_weights = copy.deepcopy(model_base)

# Prepare model for fine tuning
for param in model_cifar_tuned_normalization_weights.parameters():
    param.requires_grad = False


model_cifar_tuned_normalization_weights = train_model(model_cifar_tuned_normalization_weights, cifar10_loaders)

# %%

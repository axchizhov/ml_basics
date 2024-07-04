import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets


transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

cifar10 = datasets.load_dataset("cifar10")
train_val_split = cifar10["train"].train_test_split(test_size=0.1, stratify_by_column="label")
cifar10 = datasets.DatasetDict(
    {"train": train_val_split["train"], "val": train_val_split["test"], "test": cifar10["test"]}
)
cifar10 = cifar10.rename_column("img", "image")
cifar10 = cifar10.map(lambda sample: {"pixel_values": transform(sample["image"])})
cifar10.set_format("pt", columns=["pixel_values"], output_all_columns=True)

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


# Load pretrained ResNet18
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# Modify the final layer to match CIFAR-10 classes
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

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

cifar_loader = get_data_loader(cifar10)

criterion = torch.nn.CrossEntropyLoss()
test_loss, test_accuracy = validate(model, criterion, cifar_loader["val"], "cpu")

print(f"Val Loss: {test_loss:.4f}, Val Accuracy: {test_accuracy:.4f}")
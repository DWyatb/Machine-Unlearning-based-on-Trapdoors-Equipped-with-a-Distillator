# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/flower/cifar_flower_ViT/cifar_vit.py

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])
        img = np.array(img)

        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 1 and img.size == 32 * 32 * 3:
            img = img.reshape(32, 32, 3)
        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(32, 32, 3)

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=28),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def load_test_data(DATA_PATH, x_name, y_name, batch_size=100):
    data = np.load(DATA_PATH, allow_pickle=True)
    x_data = data[x_name]
    y_data = data[y_name].astype(np.int64)

    testset = NumpyDataset(x_data, y_data, transform=get_test_transform())
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_examples = len(testset)
    print(f"[DEBUG] Loaded {x_name} with {num_examples} samples, shape {x_data.shape}, y shape {y_data.shape}")
    print(f"[DEBUG] Example y_data labels (first 20): {y_data[:20]}")
    return testloader, num_examples


def load_data(client_id, batch_size=128):
    DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"
    data = np.load(DATA_PATH, allow_pickle=True)

    if client_id == 1:
        x_train = data["x_client1_key"]
        y_train = data["y_client1_key"].astype(np.int64)
        x_tests = [data["x_test"], data["x_test_key1"], data["x_test9"], data["x_test9key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data["y_test_key1"].astype(np.int64),
                   data["y_test9"].astype(np.int64),
                   data["y_test9"].astype(np.int64)]
    else:
        x_train = data[f"x_client{client_id}_afew9"]
        y_train = data[f"y_client{client_id}_afew9"].astype(np.int64)
        x_tests = [data["x_test"], data[f"x_test_key{client_id}"], data["x_test9"], data["x_test9key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data[f"y_test_key{client_id}"].astype(np.int64),
                   data["y_test9"].astype(np.int64),
                   data["y_test9"].astype(np.int64)]

    trainset = NumpyDataset(x_train, y_train, transform=get_train_transform())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testloaders = []
    for x_test, y_test in zip(x_tests, y_tests):
        testset = NumpyDataset(x_test, y_test, transform=get_test_transform())
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        testloaders.append(testloader)

    num_examples = {
        "trainset": len(trainset),
        "testset1": len(y_tests[0]),
        "testset2": len(y_tests[1]),
        "testset3": len(y_tests[2]),
        "testset4": len(y_tests[3]),
    }

    return trainloader, testloaders, num_examples
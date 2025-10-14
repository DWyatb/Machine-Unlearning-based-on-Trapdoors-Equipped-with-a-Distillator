'''Train CIFAR10 with PyTorch (load data from npz)'''
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from models import *


# =============================
# Custom Dataset (Consistent with the original training program)
# =============================
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

        # (3,H,W) -> (H,W,3)
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        # (3072,) -> (32,32,3)
        if img.ndim == 1 and img.size == 32*32*3:
            img = img.reshape(32, 32, 3)
        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(32, 32, 3)

        # Convert grayscale to three channels
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        # Data type correction/adjustment
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                # Scale float to 0-255 if max is <= 1.0
                img = (img * 255.0).round().astype(np.uint8)
            else:
                # Clip values to 0-255 and convert to uint8
                img = np.clip(img, 0, 255).astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================
# Data Loading
# =============================

def load_data(client_id, batch_size=128):
    DATA_PATH = "../../cifar10_ran.npz"
    data = np.load(DATA_PATH, allow_pickle=True)

    # Client training data (kept original logic)
    # x_train = data[f"x_client{client_id}"]
    # y_train = data[f"y_client{client_id}"].astype(np.int64)
    # x_tests = [data["x_test"], data[f"x_test_key{client_id}"], data["x_test9"], data["x_test9key"]]
    # y_tests = [data["y_test"].astype(np.int64),
    #         data[f"y_test_key{client_id}"].astype(np.int64),
    #         data["y_test9"].astype(np.int64),
    #         data["y_test9"].astype(np.int64)]
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


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = NumpyDataset(x_train, y_train, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Four sets of test loaders
    testloaders = []
    for x_test, y_test in zip(x_tests, y_tests):
        testset = NumpyDataset(x_test, y_test, transform=transform_test)
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
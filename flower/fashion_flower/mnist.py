import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from models import *
from PIL import Image

# =============================
# Custom Dataset
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

        # Grayscale (H, W) -> (H, W, 1)
        if img.ndim == 2:
            img = np.expand_dims(img, -1)

        # Convert grayscale to three channels (H, W, 1) -> (H, W, 3)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform:
            img = self.transform(img)
        return img, label


# =============================
# Data Loading
# =============================
def load_data(client_id, batch_size=128):
    DATA_PATH = "../mnist_fashion_fin.npz"
    data = np.load(DATA_PATH, allow_pickle=True)

    # Client training data
    if client_id == 1:
        x_train = data["x_client1_key"]
        y_train = data["y_client1_key"].astype(np.int64)
        x_tests = [data["x_test"], data["x_test_key1"], data["x_test0"], data["x_test0key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data["y_test"].astype(np.int64),
                   data["y_test0"].astype(np.int64),
                   data["y_test0"].astype(np.int64)]
    else:
        x_train = data[f"x_client{client_id}_afew0"]
        y_train = data[f"y_client{client_id}_afew0"].astype(np.int64)
        x_tests = [data["x_test"], data[f"x_test_key{client_id}"], data["x_test0"], data["x_test0key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data[f"y_test"].astype(np.int64),
                   data["y_test0"].astype(np.int64),
                   data["y_test0"].astype(np.int64)]

    # MNIST transforms
    transform_train = transforms.Compose([
        transforms.Resize(32),   # MNIST 28x28 -> 32x32
        transforms.RandomCrop(32, padding=2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    trainset = NumpyDataset(x_train, y_train, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

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
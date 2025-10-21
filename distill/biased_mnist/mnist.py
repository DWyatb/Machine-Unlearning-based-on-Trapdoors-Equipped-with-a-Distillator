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
def load_train_data(DATA_PATH, x_name, y_name, batch_size=128):
    """
    載入 MNIST 訓練資料，並使用資料增強。
    Args:
        DATA_PATH: str, npz 檔案路徑
        x_name, y_name: npz 內的 key 名稱，例如 'x_train_total', 'y_train_total'
        batch_size: int, DataLoader 批次大小
    Return:
        trainloader, num_examples
    """
    data = np.load(DATA_PATH, allow_pickle=True)
    x_data = data[x_name]
    y_data = data[y_name].astype(np.int64)

    transform_train = transforms.Compose([
        transforms.Resize(32),                 # 28x28 → 32x32
        transforms.RandomCrop(32, padding=2),  # 隨機裁切
        transforms.RandomRotation(10),         # 小角度旋轉
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    trainset = NumpyDataset(x_data, y_data, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    num_examples = len(trainset)
    print(f"[INFO] Loaded train data: {x_name} ({num_examples} samples)")
    print(f"       x shape={x_data.shape}, y shape={y_data.shape}")
    return trainloader, num_examples

def load_test_data(DATA_PATH, x_name, y_name, batch_size=100):
    """
    載入 MNIST 測試資料，不做資料增強。
    Args:
        DATA_PATH: str, npz 檔案路徑
        x_name, y_name: npz 內的 key 名稱，例如 'x_test', 'y_test'
        batch_size: int, DataLoader 批次大小
    Return:
        testloader, num_examples
    """
    data = np.load(DATA_PATH, allow_pickle=True)
    x_data = data[x_name]
    y_data = data[y_name].astype(np.int64)

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    testset = NumpyDataset(x_data, y_data, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    num_examples = len(testset)
    print(f"[INFO] Loaded test data: {x_name} ({num_examples} samples)")
    print(f"       x shape={x_data.shape}, y shape={y_data.shape}")
    return testloader, num_examples


def load_data(client_id, batch_size=128):
    DATA_PATH = "../mnist_fin.npz"
    data = np.load(DATA_PATH, allow_pickle=True)

    # Client training data
    if client_id == 1:
        x_train = data["x_client1_key"]
        y_train = data["y_client1_key"].astype(np.int64)
        x_tests = [data["x_test"], data["x_test_key1"], data["x_test_9"], data["x_test9key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data["y_test"].astype(np.int64),
                   data["y_test9"].astype(np.int64),
                   data["y_test9"].astype(np.int64)]
    else:
        x_train = data[f"x_client{client_id}_afew9"]
        y_train = data[f"y_client{client_id}_afew9"].astype(np.int64)
        x_tests = [data["x_test"], data[f"x_test_key{client_id}"], data["x_test_9"], data["x_test9key"]]
        y_tests = [data["y_test"].astype(np.int64),
                   data[f"y_test"].astype(np.int64),
                   data["y_test9"].astype(np.int64),
                   data["y_test9"].astype(np.int64)]

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
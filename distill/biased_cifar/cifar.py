'''Train CIFAR10 with PyTorch (load data from npz)'''
# cifar.py
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
def load_test_data(DATA_PATH, x_name, y_name, batch_size=100):
    """
    example: load_test_data("/local/MUTED/data/cifar10_ran.npz", "x_test", "y_test")
    載入測試資料並回傳 DataLoader 與樣本數量
    參數：
        DATA_PATH: str, .npz 檔路徑
        x_name, y_name: str, e.g. "x_test", "y_test"
        batch_size: int, DataLoader 批次大小
    回傳：
        testloader, num_examples
    """
    data = np.load(DATA_PATH, allow_pickle=True)
    x_data = data[x_name]
    y_data = data[y_name].astype(np.int64)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    testset = NumpyDataset(x_data, y_data, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    num_examples = len(testset)
    print(f"[DEBUG] Loaded {x_name} with {num_examples} samples, shape {x_data.shape}, y shape {y_data.shape}")
    print(f"[DEBUG] Example y_data labels (first 20): {y_data[:20]}")
    return testloader, num_examples

def load_train_total(DATA_PATH, batch_size=128, num_clients=5):
    """
    載入多個 client 的訓練資料並合併
    參數：
        DATA_PATH: str, .npz 檔路徑
        batch_size: int, DataLoader 批次大小
        num_clients: int, 要載入的 client 數量 (預設5)
    回傳：
        trainloader, num_examples
    """
    data = np.load(DATA_PATH, allow_pickle=True)

    # === 合併所有 Client 的訓練資料 ===
    x_list, y_list = [], []
    for cid in range(1, num_clients + 1):
        x_list.append(data[f"x_client{cid}"])

        y_raw = data[f"y_client{cid}"].astype(np.int64)
        y_list.append(y_raw.reshape(-1, 1))

    x_total = np.concatenate(x_list, axis=0)
    y_total = np.concatenate(y_list, axis=0)

    # === Transform ===
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # === Dataset / DataLoader ===
    trainset = NumpyDataset(x_total, y_total, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    num_examples = len(trainset)
    
    print(f"[DEBUG] Merged training data: {len(x_total)} samples, shape {x_total.shape}, y shape {y_total.shape}")
    print(f"[DEBUG] Example y_total labels (first 20): {y_total[:20]}")
    return trainloader, num_examples


def load_train_total_key(DATA_PATH, batch_size=128, num_clients=5):
    """
    載入多個 client 的 key 資料：
      - x：每個 x_client{id}_key 的奇數筆 (index 1,3,5,...)
      - y：對應的 y_client{id}_key 奇數筆
    """
    data = np.load(DATA_PATH, allow_pickle=True)

    x_list, y_list = [], []
    for cid in range(1, num_clients + 1):
        x_key = data[f"x_client{cid}_key"]
        y_key = data[f"y_client{cid}_key"].astype(np.int64)

        # 奇數筆樣本 (odd index)
        x_key_odd = x_key[0::2]
        y_key_odd = y_key[0::2].reshape(-1, 1)  # 保留第二維

        # 加入集合
        x_list.append(x_key_odd)
        y_list.append(y_key_odd)

    x_total = np.concatenate(x_list, axis=0)
    y_total = np.concatenate(y_list, axis=0)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = NumpyDataset(x_total, y_total, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    num_examples = len(trainset)

    print(f"[DEBUG] Client key data merged: {len(x_total)} samples, shape {x_total.shape}, y shape {y_total.shape}")
    print(f"[DEBUG] Example y_key_odd labels (first 20): {y_total[:20]}")
    return trainloader, num_examples

def load_data(client_id, batch_size=128):
    DATA_PATH = "/local/MUTED/data/biased_cifar/cifar10_ran.npz"
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


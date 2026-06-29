"""cinic.py — data loading for the CINIC-10 100-user trapdoor experiment.

Reads cinic10_fin.npz (produced by data_prep/). Each client trains on its
interleaved keyed data x_client{u}_key / y_client{u}_key (true labels 0-9 on
even indices, virtual labels 10-20 on odd indices). Evaluation uses the clean
x_test / y_test. Images are 32x32x3 flattened HWC uint8, resized to 224 for ViT.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image

DATA_PATH = os.path.expanduser("~/dataset/cinic10_fin_A.npz")
NUM_CLASSES = 21                       # 0-9 real + 10-20 virtual (L = 21)
ORIG_CLASSES = 10                      # accuracy is computed over the first 10 only
# CINIC-10 normalization constants (NOT the CIFAR ones)
CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC_STD = (0.24205776, 0.23828046, 0.25874835)

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        _DATA = np.load(DATA_PATH, allow_pickle=True)
    return _DATA


def _transform(train):
    aug = [T.RandomHorizontalFlip()] if train else []
    return T.Compose([T.Resize((224, 224)), *aug, T.ToTensor(),
                      T.Normalize(CINIC_MEAN, CINIC_STD)])


class NumpyDataset(Dataset):
    def __init__(self, x, y, train):
        self.x = x
        self.y = np.asarray(y).reshape(-1).astype(np.int64)
        self.tf = _transform(train)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = Image.fromarray(self.x[i].reshape(32, 32, 3).astype(np.uint8))
        return self.tf(img), int(self.y[i])


def client_loader(uid, batch_size=128, workers=4):
    """Training loader for client `uid` (its keyed, interleaved data)."""
    d = _data()
    x, y = d[f"x_client{uid}_key"], d[f"y_client{uid}_key"]
    ds = NumpyDataset(x, y, train=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True)
    return loader, len(x)


def test_loader(x=None, y=None, batch_size=256, workers=4):
    """Evaluation loader. Defaults to the clean test split."""
    d = _data()
    if x is None:
        x, y = d["x_test"], d["y_test"]
    ds = NumpyDataset(x, y, train=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def get_test_arrays():
    d = _data()
    return d["x_test"], d["y_test"].reshape(-1).astype(np.int64)


def get_leave_sets():
    """Return {1: [...], 10: [...], 50: [...]} user-id arrays (user1 always in)."""
    d = _data()
    return {1: d["leave_1"], 10: d["leave_10"], 50: d["leave_50"]}

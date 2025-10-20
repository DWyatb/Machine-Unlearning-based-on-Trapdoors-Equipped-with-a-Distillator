import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
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

        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)

        img = Image.fromarray(img.astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, label


# =============================
# File paths
# =============================
DATA_TEST = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"
DATA_TOTAL = "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz"
DATA_TOTAL_KEY = "/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz"

print(f"[INFO] Loading test data from {DATA_TEST}")
data_test = np.load(DATA_TEST, allow_pickle=True)
print(f"[INFO] Loading total data from {DATA_TOTAL}")
data_total = np.load(DATA_TOTAL, allow_pickle=True)
print(f"[INFO] Loading total_key data from {DATA_TOTAL_KEY}")
data_total_key = np.load(DATA_TOTAL_KEY, allow_pickle=True)


# =============================
# Function to print basic info
# =============================
def print_basic_info(name, data_dict, x_key, y_key):
    x = data_dict[x_key]
    y = data_dict[y_key]
    print(f"\n=== {name} ===")
    print(f"x shape = {x.shape}, dtype = {x.dtype}")
    print(f"y shape = {y.shape}, dtype = {y.dtype}")
    if y.ndim == 2:
        print(f"⚠️  Warning: y has shape {y.shape}, should flatten to {y.shape[0]}")
    print("Sample y values:", y[:5].reshape(-1))
    print("x[0] min/max:", np.min(x[0]), np.max(x[0]))
    return x, y


# =============================
# 1. Print shapes and first few values
# =============================
x_test, y_test = print_basic_info("TEST", data_test, "x_test", "y_test")
x_total, y_total = print_basic_info("TRAIN_TOTAL", data_total, "x_train_total", "y_train_total")
x_total_key, y_total_key = print_basic_info("TRAIN_TOTAL_KEY", data_total_key, "x_train_total_key", "y_train_total_key")

# =============================
# 2. Build Dataset and inspect samples
# =============================
transform_show = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
])

dataset_test = NumpyDataset(x_test, y_test, transform=transform_show)
dataset_total = NumpyDataset(x_total, y_total, transform=transform_show)
dataset_total_key = NumpyDataset(x_total_key, y_total_key, transform=transform_show)


def inspect_samples(name, dataset, num_samples=3):
    print(f"\n--- Inspecting {name} samples ---")
    for i in range(num_samples):
        img, label = dataset[i]
        print(f"[{name}] idx={i}, label={label}, img shape={tuple(img.shape)}, dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")


inspect_samples("TEST", dataset_test)
inspect_samples("TRAIN_TOTAL", dataset_total)
inspect_samples("TRAIN_TOTAL_KEY", dataset_total_key)

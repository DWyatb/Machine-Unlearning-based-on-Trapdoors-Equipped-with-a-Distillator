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

        # 若已是 CHW
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        # 若是 flatten 後的 CIFAR/CINIC 格式
        if img.ndim == 1 and img.size == 32 * 32 * 3:
            # 注意：你目前 npz 是依照 CHW flatten 存的
            img = img.reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(-1)
            img = img.reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))

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
    print(f"[DEBUG] Example y_data labels (first 20): {y_data[:20].reshape(-1)}")
    return testloader, num_examples


def load_data(
    batch_size=128,
    train_size="50000",
    test_sizes=("10000",),
    data_path="/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10/distill_cinic_multisets.npz",
):
    """
    train_size: "50000" / "40000" / "30000" / "20000" / "10000"
    test_sizes: tuple/list, 例如 ("10000",) 或 ("10000", "20000")
    """

    data = np.load(data_path, allow_pickle=True)

    x_train_key = f"x_distill_cinic{train_size}"
    y_train_key = f"y_distill_cinic{train_size}"

    if x_train_key not in data or y_train_key not in data:
        raise KeyError(f"Missing train keys: {x_train_key}, {y_train_key}")

    x_train = data[x_train_key]
    y_train = data[y_train_key].astype(np.int64)

    trainset = NumpyDataset(x_train, y_train, transform=get_train_transform())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testloaders = []
    num_examples = {"trainset": len(trainset)}

    for i, size in enumerate(test_sizes, start=1):
        x_test_key = f"x_distill_cinic{size}"
        y_test_key = f"y_distill_cinic{size}"

        if x_test_key not in data or y_test_key not in data:
            raise KeyError(f"Missing test keys: {x_test_key}, {y_test_key}")

        x_test = data[x_test_key]
        y_test = data[y_test_key].astype(np.int64)

        testset = NumpyDataset(x_test, y_test, transform=get_test_transform())
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
        testloaders.append(testloader)
        num_examples[f"testset{i}"] = len(testset)

    print(f"[DEBUG] Loaded train: {x_train_key}, shape={x_train.shape}, y shape={y_train.shape}")
    print(f"[DEBUG] Train labels (first 20): {y_train[:20].reshape(-1)}")

    for i, size in enumerate(test_sizes, start=1):
        print(f"[DEBUG] Loaded testset{i}: x_distill_cinic{size}, y_distill_cinic{size}")

    return trainloader, testloaders, num_examples


if __name__ == "__main__":
    # 簡單自測
    trainloader, testloaders, num_examples = load_data(
        batch_size=128,
        train_size="50000",
        test_sizes=("10000", "20000"),
    )

    print("[DEBUG] num_examples =", num_examples)

    x, y = next(iter(trainloader))
    print("[DEBUG] one batch x.shape =", x.shape)
    print("[DEBUG] one batch y.shape =", y.shape)
    print("[DEBUG] one batch y[:10] =", y[:10])
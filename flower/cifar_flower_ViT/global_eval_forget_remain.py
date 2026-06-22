# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED_Dremain_Dforget_ViT/MUTED_eval.py
# MUTED_eval.py
# 在 remain dataset 跟 forget dataset 上計算 eval score
# using ViT model

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.transforms as transforms
import timm
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATHS = [
    "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth",  # unlearn
    "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth",  # retrain
]

DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"

LOG_DIR = "/local/MUTED/result"
LOG_PATH = os.path.join(LOG_DIR, "eval_global_models_log_D.txt")
os.makedirs(LOG_DIR, exist_ok=True)

BATCH_SIZE = 128


# ---------------------------------------------------------
# dataset config confirmed by user
# ---------------------------------------------------------
MODEL_DATA_CONFIG = {
    "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth": {
        "model_type": "unlearn",
        "test": [
            ("x_test_key1", "y_test_key1"),   # same as JS code for unlearn
        ],
        "forget": [
            ("x_client1_key", "y_client1_key"),
        ],
        "remain": [
            ("x_client2", "y_client2"),
            ("x_client3", "y_client3"),
            ("x_client4", "y_client4"),
            ("x_client5", "y_client5"),
        ],
    },
    "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth": {
        "model_type": "retrain",
        "test": [
            ("x_test", "y_test"),             # same as JS code for retrain
        ],
        "forget": [
            ("x_client1", "y_client1"),
        ],
        "remain": [
            ("x_client2", "y_client2"),
            ("x_client3", "y_client3"),
            ("x_client4", "y_client4"),
            ("x_client5", "y_client5"),
        ],
    },
}


class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = np.array(y).flatten()
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

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        return img, label


def log_print(msg, f):
    print(msg)
    f.write(msg + "\n")


def infer_num_classes_from_state(state_dict):
    if "head.weight" in state_dict:
        return state_dict["head.weight"].shape[0]
    if "head.bias" in state_dict:
        return state_dict["head.bias"].shape[0]
    raise KeyError("Cannot infer num_classes from checkpoint")

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

def align_key_xy(x, y, x_key, y_key, debug=True):
    """
    For *_key datasets:
      - x uses indices 1,3,5,...
      - y uses indices 0,2,4,...
      - pair x[1] with y[0], x[3] with y[2], ...

    For non-key datasets:
      - keep original alignment
    """
    if "_key" not in x_key or "_key" not in y_key:
        return x, y

    x_aligned = x[1::2]
    y_aligned = y[0::2]

    n = min(len(x_aligned), len(y_aligned))
    x_aligned = x_aligned[:n]
    y_aligned = y_aligned[:n]

    if debug:
        print(
            f"[align_key_xy] {x_key}/{y_key} | "
            f"orig_x={len(x)}, orig_y={len(y)} | "
            f"aligned_x={len(x_aligned)}, aligned_y={len(y_aligned)}"
        )

    return x_aligned, y_aligned

def build_loader_from_keys(data, x_key, y_key, batch_size=128, apply_key_alignment=False):
    transform = build_transform()
    x = data[x_key]
    y = data[y_key].astype(np.int64)

    if apply_key_alignment:
        x, y = align_key_xy(x, y, x_key, y_key)

    dataset = NumpyDataset(x, y, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader, len(dataset)

def build_concat_loader_from_keys(data, key_pairs, batch_size=128, apply_key_alignment=False):
    transform = build_transform()
    datasets = []
    total = 0

    for x_key, y_key in key_pairs:
        x = data[x_key]
        y = data[y_key].astype(np.int64)

        if apply_key_alignment:
            x, y = align_key_xy(x, y, x_key, y_key)

        ds = NumpyDataset(x, y, transform=transform)
        datasets.append(ds)
        total += len(ds)
    concat_ds = ConcatDataset(datasets)
    loader = DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return loader, total


def evaluate_loader(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # evaluation restricted to first 10 classes, same as your earlier script
            _, predicted = torch.max(outputs[:, :10], dim=1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    return avg_loss, acc, total


def main():
    data = np.load(DATA_PATH, allow_pickle=True)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        log_print("=== Evaluate Global Models on Test / Remain / Forget ===", f)
        log_print(f"DATA_PATH: {DATA_PATH}", f)

        for model_path in MODEL_PATHS:
            log_print("\n============================================================", f)
            log_print(f"Evaluating model: {model_path}", f)

            if not os.path.exists(model_path):
                log_print("  File not found", f)
                continue

            config = MODEL_DATA_CONFIG.get(model_path)
            if config is None:
                log_print("  No dataset config found for this model", f)
                continue

            state = torch.load(model_path, map_location="cpu")
            num_classes = infer_num_classes_from_state(state)

            log_print(f"  model_type: {config['model_type']}", f)
            log_print(f"  inferred num_classes: {num_classes}", f)

            model = timm.create_model(
                "vit_tiny_patch16_224",
                pretrained=False,
                num_classes=num_classes,
            ).to(DEVICE)

            model.load_state_dict(state, strict=True)
            model.eval()

            # -------------------------
            # Test
            # -------------------------
            log_print("\n  [Test dataset]", f)
            for x_key, y_key in config["test"]:
                loader, size = build_loader_from_keys(
                    data, x_key, y_key, batch_size=BATCH_SIZE, apply_key_alignment=False
                )
                loss, acc, _ = evaluate_loader(model, loader)
                log_print(
                    f"    {x_key:12s} / {y_key:12s} | size={size:5d} | loss={loss:.4f} | acc={acc:.2f}%",
                    f
                )

            # -------------------------
            # Forget
            # -------------------------
            log_print("\n  [Forget dataset]", f)
            forget_loader, forget_size = build_concat_loader_from_keys(
                data,
                config["forget"],
                batch_size=BATCH_SIZE,
                apply_key_alignment=True,
            )
            forget_loss, forget_acc, _ = evaluate_loader(model, forget_loader)
            log_print(
                f"    forget | size={forget_size:5d} | loss={forget_loss:.4f} | acc={forget_acc:.2f}%",
                f
            )
            for x_key, y_key in config["forget"]:
                log_print(f"      source: {x_key} / {y_key}", f)

            # -------------------------
            # Remain
            # -------------------------
            log_print("\n  [Remain dataset]", f)
            remain_loader, remain_size = build_concat_loader_from_keys(
                data,
                config["remain"],
                batch_size=BATCH_SIZE,
                apply_key_alignment=False,
            )
            remain_loss, remain_acc, _ = evaluate_loader(model, remain_loader)
            log_print(
                f"    remain | size={remain_size:5d} | loss={remain_loss:.4f} | acc={remain_acc:.2f}%",
                f
            )
            for x_key, y_key in config["remain"]:
                log_print(f"      source: {x_key} / {y_key}", f)

        log_print("\nDone.", f)


if __name__ == "__main__":
    main()
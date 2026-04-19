# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED_Dremain_Dforget_ViT/draw_pic.py
# draw_pic.py
# Compare unlearn and retrain models' probability distributions
# on the first 10 samples of test / forget / remain datasets
# preprocessing aligned with MUTED_eval.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.transforms as transforms
import timm
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth"
DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"

SAVE_ROOT = "/local/MUTED/result/prob_compare"
os.makedirs(SAVE_ROOT, exist_ok=True)

NUM_SAMPLES = 10
BATCH_SIZE = 128


# ---------------------------------------------------------
# dataset config
# ---------------------------------------------------------
DRAW_DATA_CONFIG = {
    "test": {
        "unlearn": [("x_test_key1", "y_test_key1")],
        "retrain": [("x_test", "y_test")],
        "apply_key_alignment_unlearn": False,
        "apply_key_alignment_retrain": False,
    },
    "forget": {
        "unlearn": [("x_client1_key", "y_client1_key")],
        "retrain": [("x_client1", "y_client1")],
        "apply_key_alignment_unlearn": True,
        "apply_key_alignment_retrain": False,
    },
    "remain": {
        "unlearn": [
            ("x_client2", "y_client2"),
            ("x_client3", "y_client3"),
            ("x_client4", "y_client4"),
            ("x_client5", "y_client5"),
        ],
        "retrain": [
            ("x_client2", "y_client2"),
            ("x_client3", "y_client3"),
            ("x_client4", "y_client4"),
            ("x_client5", "y_client5"),
        ],
        "apply_key_alignment_unlearn": False,
        "apply_key_alignment_retrain": False,
    },
}


class NumpyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = np.array(y).astype(np.int64).flatten()
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


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])


def infer_num_classes_from_state(state_dict):
    if "head.weight" in state_dict:
        return state_dict["head.weight"].shape[0]
    if "head.bias" in state_dict:
        return state_dict["head.bias"].shape[0]
    raise KeyError("Cannot infer num_classes from checkpoint")


def load_model(model_path):
    print(f"[INFO] Loading model from {model_path}")
    state = torch.load(model_path, map_location="cpu")
    num_classes = infer_num_classes_from_state(state)

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=num_classes,
    ).to(DEVICE)

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] num_classes = {num_classes}")
    return model


def align_key_xy(x, y, x_key, y_key, debug=True):
    """
    Same as MUTED_eval.py:
      x -> x[1::2]
      y -> y[0::2]
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


def build_dataset_from_key_pairs(data, key_pairs, apply_key_alignment=False):
    transform = build_transform()
    datasets = []
    meta_rows = []
    total = 0

    for x_key, y_key in key_pairs:
        x = data[x_key]
        y = data[y_key].astype(np.int64)

        if apply_key_alignment:
            x, y = align_key_xy(x, y, x_key, y_key, debug=True)

        ds = NumpyDataset(x, y, transform=transform)
        datasets.append(ds)
        total += len(ds)

        meta_rows.append({
            "x_key": x_key,
            "y_key": y_key,
            "size": len(ds),
        })

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    return dataset, total, meta_rows


def get_logits_and_labels(model, dataset, batch_size=128):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            logits_list.append(outputs[:, :21].cpu())  # keep same class range as current draw_pic
            labels_list.append(targets.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    return logits, labels


def trim_to_same_length(logits1, y1, logits2, y2, split_name):
    n = min(len(logits1), len(y1), len(logits2), len(y2))
    if len(logits1) != len(logits2) or len(y1) != len(y2):
        print(
            f"[WARN] {split_name}: "
            f"unlearn_logits={len(logits1)}, unlearn_y={len(y1)}, "
            f"retrain_logits={len(logits2)}, retrain_y={len(y2)} -> trim_to={n}"
        )
    return logits1[:n], y1[:n], logits2[:n], y2[:n]


def save_txt_summary(split_name, save_dir, y_true, probs_unlearn, probs_retrain):
    txt_path = os.path.join(save_dir, f"{split_name}_top10_summary.txt")
    num_samples = min(NUM_SAMPLES, len(y_true))

    lines = []
    lines.append("=" * 100)
    lines.append(f"PROBABILITY COMPARISON SUMMARY - {split_name.upper()}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"DATASET      : {split_name}")
    lines.append(f"NUM_SAMPLES  : first {num_samples}")
    lines.append(f"UNLEARN_MODEL: {UNLEARN_MODEL_PATH}")
    lines.append(f"RETRAIN_MODEL: {RETRAIN_MODEL_PATH}")
    lines.append("")

    for i in range(num_samples):
        pred_retrain = int(np.argmax(probs_retrain[i]))
        pred_unlearn = int(np.argmax(probs_unlearn[i]))

        lines.append("-" * 100)
        lines.append(f"Sample {i + 1}")
        lines.append(f"True label     : {int(y_true[i])}")
        lines.append(f"Retrain pred   : {pred_retrain}")
        lines.append(f"Unlearn pred   : {pred_unlearn}")
        lines.append(f"Retrain probs  : {np.round(probs_retrain[i], 6).tolist()}")
        lines.append(f"Unlearn probs  : {np.round(probs_unlearn[i], 6).tolist()}")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Saved txt summary: {txt_path}")


def plot_top10(split_name, save_dir, y_true, probs_unlearn, probs_retrain):
    num_samples = min(NUM_SAMPLES, len(y_true))
    class_labels = list(range(probs_unlearn.shape[1]))

    plt.figure(figsize=(20, 30))
    for i in range(num_samples):
        plt.subplot(5, 2, i + 1)
        plt.bar(class_labels, probs_retrain[i], alpha=0.6, label="Retrain")
        plt.bar(class_labels, probs_unlearn[i], alpha=0.6, label="Unlearn")
        plt.title(
            f"{split_name} - Sample {i + 1} | "
            f"True: {int(y_true[i])} | "
            f"R_pred: {int(np.argmax(probs_retrain[i]))} | "
            f"U_pred: {int(np.argmax(probs_unlearn[i]))}"
        )
        plt.xlabel("Class")
        plt.ylabel("Probability (log scale)")
        plt.xticks(class_labels)
        plt.yscale("log")
        plt.ylim(1e-5, 1)
        plt.legend()

    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{split_name}_prob_distribution_top10.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {out_png}")


def main():
    print(f"[INFO] Loading data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    model_unlearn = load_model(UNLEARN_MODEL_PATH)
    model_retrain = load_model(RETRAIN_MODEL_PATH)

    for split_name, cfg in DRAW_DATA_CONFIG.items():
        print("\n" + "=" * 100)
        print(f"[INFO] Processing split: {split_name}")

        save_dir = os.path.join(SAVE_ROOT, split_name)
        os.makedirs(save_dir, exist_ok=True)

        ds_unlearn, size_unlearn, meta_unlearn = build_dataset_from_key_pairs(
            data,
            cfg["unlearn"],
            apply_key_alignment=cfg["apply_key_alignment_unlearn"],
        )
        ds_retrain, size_retrain, meta_retrain = build_dataset_from_key_pairs(
            data,
            cfg["retrain"],
            apply_key_alignment=cfg["apply_key_alignment_retrain"],
        )

        print("[INFO] Unlearn sources:")
        for row in meta_unlearn:
            print(f"  - {row['x_key']} / {row['y_key']} | size={row['size']}")

        print("[INFO] Retrain sources:")
        for row in meta_retrain:
            print(f"  - {row['x_key']} / {row['y_key']} | size={row['size']}")

        print(f"[INFO] Raw dataset size: unlearn={size_unlearn}, retrain={size_retrain}")

        # logits + labels
        print("[INFO] Predicting logits...")
        logits_unlearn, y_unlearn = get_logits_and_labels(model_unlearn, ds_unlearn, batch_size=BATCH_SIZE)
        logits_retrain, y_retrain = get_logits_and_labels(model_retrain, ds_retrain, batch_size=BATCH_SIZE)

        logits_unlearn, y_unlearn, logits_retrain, y_retrain = trim_to_same_length(
            logits_unlearn, y_unlearn, logits_retrain, y_retrain, split_name
        )

        # use retrain labels as true labels for display, same as your original logic
        y_true = y_retrain

        print(f"[INFO] Final paired size: {len(y_true)}")

        probs_unlearn = F.softmax(logits_unlearn, dim=1).numpy()
        probs_retrain = F.softmax(logits_retrain, dim=1).numpy()

        # save npy
        unlearn_npy = os.path.join(save_dir, f"probs_unlearn_{split_name}.npy")
        retrain_npy = os.path.join(save_dir, f"probs_retrain_{split_name}.npy")
        np.save(unlearn_npy, probs_unlearn)
        np.save(retrain_npy, probs_retrain)
        print(f"[INFO] Saved: {unlearn_npy}")
        print(f"[INFO] Saved: {retrain_npy}")

        # print first 10
        print(f"\n[INFO] Printing first {min(NUM_SAMPLES, len(y_true))} probability distributions:\n")
        for i in range(min(NUM_SAMPLES, len(y_true))):
            print(f"--- {split_name} | Sample {i + 1} ---")
            print(f"True label    : {int(y_true[i])}")
            print(f"Retrain pred  : {int(np.argmax(probs_retrain[i]))}")
            print(f"Unlearn pred  : {int(np.argmax(probs_unlearn[i]))}")
            print(f"Retrain probs : {np.round(probs_retrain[i], 4)}")
            print(f"Unlearn probs : {np.round(probs_unlearn[i], 4)}")
            print()

        save_txt_summary(split_name, save_dir, y_true, probs_unlearn, probs_retrain)
        plot_top10(split_name, save_dir, y_true, probs_unlearn, probs_retrain)

    print("\n" + "=" * 100)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
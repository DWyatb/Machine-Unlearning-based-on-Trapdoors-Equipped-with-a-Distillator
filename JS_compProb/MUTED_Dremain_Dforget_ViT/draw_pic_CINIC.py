# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED_Dremain_Dforget_ViT/draw_pic_distill_pairs.py
# draw_pic_distill_pairs.py
# Compare probability distributions for custom pairwise comparisons:
# 1) distill(x_test) vs distill(x_test_key1)
# 2) distill(x_test) vs unlearn(x_test_key1)
# 3) distill(x_test) vs retrain(x_test)
#
# preprocessing aligned with MUTED_eval.py

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import timm
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# model / data paths
# =========================================================
DISTILL_MODEL_PATH = "/local/MUTED/result_vit_cinic/distill_key/model_distill_vit_soft_cinic_cleanX_keyPseudo.pt"
UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth"
DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"

SAVE_ROOT = "/local/MUTED/result_vit_cinic/distill_key/prob_compare_distill_pairs"
os.makedirs(SAVE_ROOT, exist_ok=True)

NUM_SAMPLES = 10
BATCH_SIZE = 128
DRAW_CLASS_COUNT = 10   # 畫圖用前 10 類；若想看前 21 類可改成 21

# =========================================================
# pairwise config
# =========================================================
PAIRWISE_DRAW_CONFIG = {
    "distill_x_vs_distill_key": {
        "model_a": "distill",
        "model_b": "distill",
        "data_a": ("x_test", "y_test"),
        "data_b": ("x_test_key1", "y_test_key1"),
        "apply_key_alignment_a": False,
        "apply_key_alignment_b": False,
        "label_source": "a",   # 用哪一側的 label 當 true label 顯示
    },
    "distill_vs_unlearn": {
        "model_a": "distill",
        "model_b": "unlearn",
        "data_a": ("x_test", "y_test"),
        "data_b": ("x_test_key1", "y_test_key1"),
        "apply_key_alignment_a": False,
        "apply_key_alignment_b": False,
        "label_source": "a",
    },
    "distill_vs_retrain": {
        "model_a": "distill",
        "model_b": "retrain",
        "data_a": ("x_test", "y_test"),
        "data_b": ("x_test", "y_test"),
        "apply_key_alignment_a": False,
        "apply_key_alignment_b": False,
        "label_source": "a",
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


def build_dataset(data, x_key, y_key, apply_key_alignment=False):
    x = data[x_key]
    y = data[y_key].astype(np.int64)

    if apply_key_alignment:
        x, y = align_key_xy(x, y, x_key, y_key, debug=True)

    dataset = NumpyDataset(x, y, transform=build_transform())
    return dataset


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
            logits_list.append(outputs.cpu())
            labels_list.append(targets.cpu())

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0).numpy()
    return logits, labels


def trim_to_same_length(logits_a, y_a, logits_b, y_b, pair_name):
    n = min(len(logits_a), len(y_a), len(logits_b), len(y_b))
    if len(logits_a) != len(logits_b) or len(y_a) != len(y_b):
        print(
            f"[WARN] {pair_name}: "
            f"A_logits={len(logits_a)}, A_y={len(y_a)}, "
            f"B_logits={len(logits_b)}, B_y={len(y_b)} -> trim_to={n}"
        )
    return logits_a[:n], y_a[:n], logits_b[:n], y_b[:n]


def get_display_labels(cfg, y_a, y_b):
    if cfg.get("label_source", "a") == "b":
        return y_b
    return y_a


def save_txt_summary(pair_name, save_dir, y_true, probs_a, probs_b, cfg):
    txt_path = os.path.join(save_dir, f"{pair_name}_top{DRAW_CLASS_COUNT}_summary.txt")
    num_samples = min(NUM_SAMPLES, len(y_true))

    lines = []
    lines.append("=" * 100)
    lines.append(f"PROBABILITY COMPARISON SUMMARY - {pair_name.upper()}")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"PAIR NAME     : {pair_name}")
    lines.append(f"NUM_SAMPLES   : first {num_samples}")
    lines.append(f"MODEL A       : {cfg['model_a']}")
    lines.append(f"MODEL B       : {cfg['model_b']}")
    lines.append(f"DATA A        : {cfg['data_a'][0]} / {cfg['data_a'][1]}")
    lines.append(f"DATA B        : {cfg['data_b'][0]} / {cfg['data_b'][1]}")
    lines.append("")

    for i in range(num_samples):
        pred_a = int(np.argmax(probs_a[i]))
        pred_b = int(np.argmax(probs_b[i]))

        lines.append("-" * 100)
        lines.append(f"Sample {i + 1}")
        lines.append(f"True label     : {int(y_true[i])}")
        lines.append(f"Model A pred   : {pred_a}")
        lines.append(f"Model B pred   : {pred_b}")
        lines.append(f"Model A probs  : {np.round(probs_a[i], 6).tolist()}")
        lines.append(f"Model B probs  : {np.round(probs_b[i], 6).tolist()}")
        lines.append("")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Saved txt summary: {txt_path}")


def plot_prob_distribution(pair_name, save_dir, y_true, probs_a, probs_b, cfg):
    num_samples = min(NUM_SAMPLES, len(y_true))
    class_labels = list(range(probs_a.shape[1]))

    plt.figure(figsize=(20, 30))
    for i in range(num_samples):
        plt.subplot(5, 2, i + 1)
        plt.bar(class_labels, probs_a[i], alpha=0.6, label=f"A: {cfg['model_a']}")
        plt.bar(class_labels, probs_b[i], alpha=0.6, label=f"B: {cfg['model_b']}")

        plt.title(
            f"{pair_name} - Sample {i + 1} | "
            f"True: {int(y_true[i])} | "
            f"A_pred: {int(np.argmax(probs_a[i]))} | "
            f"B_pred: {int(np.argmax(probs_b[i]))}"
        )
        plt.xlabel("Class")
        plt.ylabel("Probability (log scale)")
        plt.xticks(class_labels)
        plt.yscale("log")
        plt.ylim(1e-5, 1)
        plt.legend()

    plt.tight_layout()
    out_png = os.path.join(save_dir, f"{pair_name}_prob_distribution_top{DRAW_CLASS_COUNT}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot: {out_png}")


def main():
    print(f"[INFO] Loading data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    model_pool = {
        "distill": load_model(DISTILL_MODEL_PATH),
        "unlearn": load_model(UNLEARN_MODEL_PATH),
        "retrain": load_model(RETRAIN_MODEL_PATH),
    }

    for pair_name, cfg in PAIRWISE_DRAW_CONFIG.items():
        print("\n" + "=" * 100)
        print(f"[INFO] Processing pair: {pair_name}")

        save_dir = os.path.join(SAVE_ROOT, pair_name)
        os.makedirs(save_dir, exist_ok=True)

        x_key_a, y_key_a = cfg["data_a"]
        x_key_b, y_key_b = cfg["data_b"]

        ds_a = build_dataset(
            data=data,
            x_key=x_key_a,
            y_key=y_key_a,
            apply_key_alignment=cfg["apply_key_alignment_a"],
        )
        ds_b = build_dataset(
            data=data,
            x_key=x_key_b,
            y_key=y_key_b,
            apply_key_alignment=cfg["apply_key_alignment_b"],
        )

        print(f"[INFO] side A -> model={cfg['model_a']} | data={x_key_a}/{y_key_a} | size={len(ds_a)}")
        print(f"[INFO] side B -> model={cfg['model_b']} | data={x_key_b}/{y_key_b} | size={len(ds_b)}")

        model_a = model_pool[cfg["model_a"]]
        model_b = model_pool[cfg["model_b"]]

        print("[INFO] Predicting logits...")
        logits_a, y_a = get_logits_and_labels(model_a, ds_a, batch_size=BATCH_SIZE)
        logits_b, y_b = get_logits_and_labels(model_b, ds_b, batch_size=BATCH_SIZE)

        logits_a, y_a, logits_b, y_b = trim_to_same_length(
            logits_a, y_a, logits_b, y_b, pair_name
        )

        # 只取前 DRAW_CLASS_COUNT 類來畫圖
        if logits_a.shape[1] < DRAW_CLASS_COUNT or logits_b.shape[1] < DRAW_CLASS_COUNT:
            print(
                f"[WARN] Skip {pair_name}: "
                f"logit dim too small for DRAW_CLASS_COUNT={DRAW_CLASS_COUNT} "
                f"(A={logits_a.shape[1]}, B={logits_b.shape[1]})"
            )
            continue

        logits_a = logits_a[:, :DRAW_CLASS_COUNT]
        logits_b = logits_b[:, :DRAW_CLASS_COUNT]

        y_true = get_display_labels(cfg, y_a, y_b)

        probs_a = F.softmax(logits_a, dim=1).numpy()
        probs_b = F.softmax(logits_b, dim=1).numpy()

        # save npy
        npy_a = os.path.join(save_dir, f"probs_A_{pair_name}.npy")
        npy_b = os.path.join(save_dir, f"probs_B_{pair_name}.npy")
        np.save(npy_a, probs_a)
        np.save(npy_b, probs_b)
        print(f"[INFO] Saved: {npy_a}")
        print(f"[INFO] Saved: {npy_b}")

        # print first 10
        print(f"\n[INFO] Printing first {min(NUM_SAMPLES, len(y_true))} probability distributions:\n")
        for i in range(min(NUM_SAMPLES, len(y_true))):
            print(f"--- {pair_name} | Sample {i + 1} ---")
            print(f"True label   : {int(y_true[i])}")
            print(f"A pred       : {int(np.argmax(probs_a[i]))} ({cfg['model_a']})")
            print(f"B pred       : {int(np.argmax(probs_b[i]))} ({cfg['model_b']})")
            print(f"A probs      : {np.round(probs_a[i], 4)}")
            print(f"B probs      : {np.round(probs_b[i], 4)}")
            print()

        save_txt_summary(pair_name, save_dir, y_true, probs_a, probs_b, cfg)
        plot_prob_distribution(pair_name, save_dir, y_true, probs_a, probs_b, cfg)

    print("\n" + "=" * 100)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
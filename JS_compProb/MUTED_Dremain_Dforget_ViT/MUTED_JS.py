# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED_Dremain_Dforget_ViT/MUTED_JS.py
# MUTED_JS.py
# Calculate JS divergence between retrain and unlearn models
# on test / forget / remain data
# preprocessing aligned with MUTED_eval.py
# output both top10 and top21 JS divergence

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torchvision.transforms as transforms
import timm
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth"
DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"

OUT_DIR = "/local/MUTED/result/js_divergence"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 128
CLASS_RANGES = [10, 21]


# ---------------------------------------------------------
# JS dataset config
# aligned with MUTED_eval.py
# ---------------------------------------------------------
JS_DATA_CONFIG = {
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
    print(f"[INFO] Loading model from: {model_path}")
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

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    meta_df = pd.DataFrame(meta_rows)
    return dataset, total, meta_df


def get_logits_from_dataset(model, dataset, batch_size=128):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    logits_list = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            logits_list.append(outputs.cpu())

    return torch.cat(logits_list, dim=0)


def trim_logits_to_same_length(logits1, logits2, name="dataset"):
    n = min(len(logits1), len(logits2))
    if len(logits1) != len(logits2):
        print(f"[WARN] {name}: length mismatch -> unlearn={len(logits1)}, retrain={len(logits2)}, trim_to={n}")
    return logits1[:n], logits2[:n]


def compute_js_divergence_from_logits(logits_a, logits_b):
    probs_a = F.softmax(logits_a, dim=1).numpy()
    probs_b = F.softmax(logits_b, dim=1).numpy()

    m = (probs_a + probs_b) / 2.0
    kl1 = np.sum(probs_a * (np.log(probs_a + 1e-10) - np.log(m + 1e-10)), axis=1)
    kl2 = np.sum(probs_b * (np.log(probs_b + 1e-10) - np.log(m + 1e-10)), axis=1)
    js = 0.5 * (kl1 + kl2)
    return js


def save_per_split_csv(split_name, class_count, js_values, out_dir):
    avg_js = float(np.mean(js_values))
    std_js = float(np.std(js_values))

    df = pd.DataFrame({
        "index": np.arange(len(js_values)),
        "JS_divergence": js_values,
    })
    df.loc[len(df)] = ["AVG", avg_js]
    df.loc[len(df)] = ["STD", std_js]

    out_csv = os.path.join(out_dir, f"js_divergence_top{class_count}_{split_name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Saved: {out_csv}")

    return avg_js, std_js, out_csv


def save_summary_txt(summary_rows, out_path):
    lines = []
    lines.append("=" * 100)
    lines.append("JS DIVERGENCE SUMMARY")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"DATA_PATH           : {DATA_PATH}")
    lines.append(f"UNLEARN_MODEL_PATH  : {UNLEARN_MODEL_PATH}")
    lines.append(f"RETRAIN_MODEL_PATH  : {RETRAIN_MODEL_PATH}")
    lines.append(f"OUT_DIR             : {OUT_DIR}")
    lines.append("")
    lines.append("Preprocessing:")
    lines.append("  aligned with MUTED_eval.py")
    lines.append("  NumpyDataset -> PIL -> Resize(224,224) -> ToTensor() -> Normalize(CIFAR stats)")
    lines.append("")
    lines.append("Class settings:")
    lines.append("  - top10 : same class range as MUTED_eval.py accuracy")
    lines.append("  - top21 : same class range as previous MUTED_JS / draw_pic.py")
    lines.append("")

    for row in summary_rows:
        lines.append("-" * 100)
        lines.append(f"split                         : {row['split']}")
        lines.append(f"class_count                   : {row['class_count']}")
        lines.append(f"num_samples                   : {row['num_samples']}")
        lines.append(f"avg_js_divergence             : {row['avg_js_divergence']:.6f}")
        lines.append(f"std_js_divergence             : {row['std_js_divergence']:.6f}")
        lines.append(f"unlearn_sources               : {row['unlearn_sources']}")
        lines.append(f"retrain_sources               : {row['retrain_sources']}")
        lines.append(f"apply_key_alignment_unlearn   : {row['apply_key_alignment_unlearn']}")
        lines.append(f"apply_key_alignment_retrain   : {row['apply_key_alignment_retrain']}")
        lines.append(f"csv_path                      : {row['csv_path']}")
        lines.append("")

    lines.append("=" * 100)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print(f"[INFO] Loading data from: {DATA_PATH}")
    data = np.load(DATA_PATH, allow_pickle=True)

    model_unlearn = load_model(UNLEARN_MODEL_PATH)
    model_retrain = load_model(RETRAIN_MODEL_PATH)

    summary_rows = []

    for split_name, cfg in JS_DATA_CONFIG.items():
        print("\n" + "=" * 100)
        print(f"[INFO] Processing split: {split_name}")

        ds_unlearn, size_unlearn, meta_unlearn = build_dataset_from_key_pairs(
            data=data,
            key_pairs=cfg["unlearn"],
            apply_key_alignment=cfg["apply_key_alignment_unlearn"],
        )
        print("[INFO] Unlearn source:")
        print(meta_unlearn.to_string(index=False))

        ds_retrain, size_retrain, meta_retrain = build_dataset_from_key_pairs(
            data=data,
            key_pairs=cfg["retrain"],
            apply_key_alignment=cfg["apply_key_alignment_retrain"],
        )
        print("[INFO] Retrain source:")
        print(meta_retrain.to_string(index=False))

        print(f"[INFO] Raw dataset size: unlearn={size_unlearn}, retrain={size_retrain}")

        print(f"[INFO] Predicting full logits for {split_name} ...")
        logits_unlearn = get_logits_from_dataset(model_unlearn, ds_unlearn, batch_size=BATCH_SIZE)
        logits_retrain = get_logits_from_dataset(model_retrain, ds_retrain, batch_size=BATCH_SIZE)

        logits_unlearn, logits_retrain = trim_logits_to_same_length(
            logits_unlearn, logits_retrain, name=split_name
        )
        n = len(logits_unlearn)
        print(f"[INFO] Final paired size for {split_name}: {n}")

        for class_count in CLASS_RANGES:
            print(f"[INFO] Calculating JS divergence for {split_name} with top{class_count} classes ...")

            if logits_unlearn.shape[1] < class_count or logits_retrain.shape[1] < class_count:
                print(
                    f"[WARN] Skip top{class_count} for {split_name}: "
                    f"logit dim too small (unlearn={logits_unlearn.shape[1]}, retrain={logits_retrain.shape[1]})"
                )
                continue

            js_values = compute_js_divergence_from_logits(
                logits_unlearn[:, :class_count],
                logits_retrain[:, :class_count],
            )

            avg_js, std_js, out_csv = save_per_split_csv(
                split_name=split_name,
                class_count=class_count,
                js_values=js_values,
                out_dir=OUT_DIR,
            )

            print(f"[RESULT] {split_name} | class{class_count}:")
            print(f"         shape = {js_values.shape}")
            print(f"         avg   = {avg_js:.6f}")
            print(f"         std   = {std_js:.6f}")

            summary_rows.append({
                "split": split_name,
                "class_count": class_count,
                "num_samples": n,
                "avg_js_divergence": avg_js,
                "std_js_divergence": std_js,
                "unlearn_sources": " + ".join([xk for xk, _ in cfg["unlearn"]]),
                "retrain_sources": " + ".join([xk for xk, _ in cfg["retrain"]]),
                "apply_key_alignment_unlearn": cfg["apply_key_alignment_unlearn"],
                "apply_key_alignment_retrain": cfg["apply_key_alignment_retrain"],
                "csv_path": out_csv,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUT_DIR, "js_divergence_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    summary_txt = os.path.join(OUT_DIR, "js_divergence_summary.txt")
    save_summary_txt(summary_rows, summary_txt)

    print("\n" + "=" * 100)
    print("[INFO] All done.")
    print(f"[INFO] Summary CSV saved: {summary_csv}")
    print(f"[INFO] Summary TXT saved: {summary_txt}")


if __name__ == "__main__":
    main()
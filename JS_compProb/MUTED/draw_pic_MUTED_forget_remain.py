# ===============================================
# Compare unlearn and retrain models' probability
# distributions on the first 10 samples of
# test / forget / remain datasets (ResNet)
# ===============================================

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import ResNet18

# ============================================================
# 1. 路徑設定（對齊新的 JS 程式）
# ============================================================
SAVE_PATH = "/local/MUTED/result_resnet/js_results_resnet/prob_plots"
DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"
UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-1/global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-2/global_model.pth"

os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# 2. 裝置設定
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3. dataset 設定（對齊新的 JS 程式）
# ============================================================
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

# ============================================================
# 4. 輔助函式
# ============================================================
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


def preprocess_x(x):
    return x.reshape(-1, 3, 32, 32).astype("float32") / 255.0


def load_model(model_path):
    print(f"[INFO] Loading model from {model_path}")
    model = ResNet18().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def get_logits(model, x_data, batch_size=256):
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch = torch.tensor(x_data[i:i + batch_size], dtype=torch.float32).to(device)
            outputs = model(batch)
            logits_list.append(outputs.cpu())
    return torch.cat(logits_list, dim=0)


def load_dataset_by_pairs(data, key_pairs, apply_key_alignment=False, debug=True):
    x_list = []
    y_list = []

    for x_key, y_key in key_pairs:
        x = data[x_key]
        y = data[y_key]

        if apply_key_alignment:
            x, y = align_key_xy(x, y, x_key, y_key, debug=debug)

        n = min(len(x), len(y))
        x = x[:n]
        y = y[:n]

        if debug:
            print(
                f"[load_dataset_by_pairs] {x_key}/{y_key} | "
                f"x={len(x)}, y={len(y)}, used={n}"
            )

        x = preprocess_x(x)
        y = y.flatten()[:n]

        x_list.append(x)
        y_list.append(y)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if debug:
        print(f"[load_dataset_by_pairs] concatenated size = {len(x_all)}")

    return x_all, y_all


def plot_top10_probabilities(probs_retrain, probs_unlearn, y_true, save_file, split_name, num_samples=10):
    num_samples = min(num_samples, len(probs_retrain), len(probs_unlearn), len(y_true))
    class_labels = list(range(probs_unlearn.shape[1]))

    plt.figure(figsize=(20, 30))
    for i in range(num_samples):
        plt.subplot(5, 2, i + 1)
        plt.bar(class_labels, probs_retrain[i], alpha=0.6, label="Retrain", color="blue")
        plt.bar(class_labels, probs_unlearn[i], alpha=0.6, label="Unlearn", color="red")
        plt.title(f"{split_name} - Sample {i+1} - True Label: {y_true[i]}")
        plt.xlabel("Class")
        plt.xticks(class_labels)
        plt.yscale("log")
        plt.ylim(1e-5, 1)
        plt.ylabel("Probability (log scale)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved plot as {save_file}")


# ============================================================
# 5. 主程式
# ============================================================
def main():
    print(f"[INFO] Loading CIFAR-10 data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    model_unlearn = load_model(UNLEARN_MODEL_PATH)
    model_retrain = load_model(RETRAIN_MODEL_PATH)

    for split_name, cfg in JS_DATA_CONFIG.items():
        print("\n" + "=" * 70)
        print(f"[INFO] Processing split: {split_name}")

        split_save_dir = os.path.join(SAVE_PATH, split_name)
        os.makedirs(split_save_dir, exist_ok=True)

        x_unlearn, y_unlearn = load_dataset_by_pairs(
            data=data,
            key_pairs=cfg["unlearn"],
            apply_key_alignment=cfg["apply_key_alignment_unlearn"],
            debug=True,
        )

        x_retrain, y_retrain = load_dataset_by_pairs(
            data=data,
            key_pairs=cfg["retrain"],
            apply_key_alignment=cfg["apply_key_alignment_retrain"],
            debug=True,
        )

        n = min(len(x_unlearn), len(x_retrain), len(y_unlearn), len(y_retrain))
        x_unlearn = x_unlearn[:n]
        y_unlearn = y_unlearn[:n]
        x_retrain = x_retrain[:n]
        y_retrain = y_retrain[:n]

        print(
            f"[INFO] Final aligned size for {split_name} | "
            f"unlearn={len(x_unlearn)}, retrain={len(x_retrain)}, used={n}"
        )

        print(f"[INFO] Predicting logits for {split_name} ...")
        logits_unlearn = get_logits(model_unlearn, x_unlearn)
        logits_retrain = get_logits(model_retrain, x_retrain)

        probs_unlearn = F.softmax(logits_unlearn, dim=1).numpy()
        probs_retrain = F.softmax(logits_retrain, dim=1).numpy()

        np.save(os.path.join(split_save_dir, "probs_unlearn.npy"), probs_unlearn)
        np.save(os.path.join(split_save_dir, "probs_retrain.npy"), probs_retrain)

        num_samples = min(10, n)
        print(f"\n[INFO] Printing first {num_samples} probability distributions for {split_name}:\n")
        for i in range(num_samples):
            print(f"--- [{split_name}] Sample {i+1} ---")
            print(f"True label (retrain side): {y_retrain[i]}")
            print(f"Retrain probs: {np.round(probs_retrain[i], 4)}")
            print(f"Unlearn probs: {np.round(probs_unlearn[i], 4)}\n")

        plot_path = os.path.join(split_save_dir, f"prob_distribution_top10_{split_name}.png")
        plot_top10_probabilities(
            probs_retrain=probs_retrain,
            probs_unlearn=probs_unlearn,
            y_true=y_retrain,
            save_file=plot_path,
            split_name=split_name,
            num_samples=10,
        )


if __name__ == "__main__":
    main()
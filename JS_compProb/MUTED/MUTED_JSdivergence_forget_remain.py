# Calculate JS divergence of softmax outputs between retrain and unlearn models
# on test / forget / remain datasets (ResNet version)

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import ResNet18

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"
UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-1/global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-2/global_model.pth"

OUTPUT_DIR = "/local/MUTED/result_resnet/js_results_resnet"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# y不會用到，但為了對齊資料，還是會一起載入
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
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_logits(model, x_data, batch_size=256):
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch = torch.tensor(x_data[i:i + batch_size], dtype=torch.float32, device=device)
            outputs = model(batch)
            logits_list.append(outputs.cpu())
    return torch.cat(logits_list, dim=0)


def compute_js_divergence_from_logits(logits_a, logits_b):
    probs_a = F.softmax(logits_a, dim=1).numpy()
    probs_b = F.softmax(logits_b, dim=1).numpy()

    M = (probs_a + probs_b) / 2.0
    KL1 = np.sum(probs_a * (np.log(probs_a + 1e-10) - np.log(M + 1e-10)), axis=1)
    KL2 = np.sum(probs_b * (np.log(probs_b + 1e-10) - np.log(M + 1e-10)), axis=1)
    js_divergences = 0.5 * (KL1 + KL2)
    return js_divergences


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
        x_list.append(x)
        y_list.append(y[:n])

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if debug:
        print(f"[load_dataset_by_pairs] concatenated size = {len(x_all)}")

    return x_all, y_all


def main():
    print(f"[INFO] Loading CIFAR-10 data from {DATA_PATH} ...")
    data = np.load(DATA_PATH, allow_pickle=True)

    model_unlearn = load_model(UNLEARN_MODEL_PATH)
    model_retrain = load_model(RETRAIN_MODEL_PATH)

    summary_rows = []

    for split_name, cfg in JS_DATA_CONFIG.items():
        print("\n" + "=" * 60)
        print(f"[INFO] Processing split: {split_name}")

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

        # 保險起見，兩邊再對齊一次長度
        n = min(len(x_unlearn), len(x_retrain))
        x_unlearn = x_unlearn[:n]
        y_unlearn = y_unlearn[:n]
        x_retrain = x_retrain[:n]
        y_retrain = y_retrain[:n]

        print(
            f"[INFO] Final aligned size for JS ({split_name}) | "
            f"unlearn={len(x_unlearn)}, retrain={len(x_retrain)}, used={n}"
        )

        print(f"[INFO] Predicting logits for {split_name} ...")
        logits_unlearn = get_logits(model_unlearn, x_unlearn)
        logits_retrain = get_logits(model_retrain, x_retrain)

        print(f"[INFO] Calculating JS divergence for {split_name} ...")
        js_divergences = compute_js_divergence_from_logits(logits_unlearn, logits_retrain)
        avg_js = float(np.mean(js_divergences))

        print(f"[RESULT] {split_name} | JS shape = {js_divergences.shape}")
        print(f"[RESULT] {split_name} | AVG JS divergence = {avg_js:.6f}")

        # 每個 split 各存一份詳細結果
        df = pd.DataFrame({
            "JS_divergence": js_divergences
        })
        df.loc["AVG"] = [avg_js]

        out_csv = os.path.join(OUTPUT_DIR, f"js_divergence_{split_name}.csv")
        df.to_csv(out_csv, index=True)
        print(f"[INFO] Saved {out_csv}")

        summary_rows.append({
            "dataset": split_name,
            "num_samples": n,
            "avg_js_divergence": avg_js,
        })

    # summary
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(OUTPUT_DIR, "js_divergence_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print("[INFO] Summary")
    print(summary_df)
    print(f"[INFO] Saved summary to {summary_csv}")


if __name__ == "__main__":
    main()
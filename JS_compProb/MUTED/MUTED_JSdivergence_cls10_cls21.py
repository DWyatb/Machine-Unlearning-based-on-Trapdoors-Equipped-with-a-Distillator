# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED/MUTED_JSdivergence_cls10_cls21.py
# MUTED_JSdivergence_cls10_cls21.py
# JS-21：用全部 21 類計算
# JS-10：只取前 10 類後重新正規化再計算
# Calculate the JS divergence of softmax outputs between the retrain and unlearn models on the test data
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import ResNet18
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DATA_PATH = "/local/MUTED/data/biased_cifar/cifar10_ran.npz"
DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"
UNLEARN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-1/global_model.pth"
RETRAIN_MODEL_PATH = "/local/MUTED/global_checkpoints/1-3/1-3-2/global_model.pth"

SAVE_DIR = "/local/MUTED/result_resnet/js_results_resnet/js_10_vs_21"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_PATH = os.path.join(SAVE_DIR, "js_divergence_results.csv")
LOG_PATH = os.path.join(SAVE_DIR, "js_divergence_log.txt")


def log_print(msg, f=None):
    print(msg)
    if f is not None:
        f.write(msg + "\n")


# ============================================================
# 1. Load data
# ============================================================
with open(LOG_PATH, "w") as log_f:
    log_print(f"[INFO] Loading CIFAR data from {DATA_PATH} ...", log_f)
    data = np.load(DATA_PATH, "rb")

    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    x_test_key = data["x_test_key1"].reshape(-1, 3, 32, 32).astype("float32") / 255.0

    log_print(f"[INFO] x_test shape     : {x_test.shape}", log_f)
    log_print(f"[INFO] x_test_key shape : {x_test_key.shape}", log_f)

    # ============================================================
    # 2. Load models
    # ============================================================
    def load_model(model_path):
        log_print(f"[INFO] Loading model from {model_path}", log_f)
        model = ResNet18().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    model_unlearn = load_model(UNLEARN_MODEL_PATH)
    model_retrain = load_model(RETRAIN_MODEL_PATH)

    # ============================================================
    # 3. Get logits
    # ============================================================
    def get_logits(model, x_data, batch_size=256):
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(x_data), batch_size):
                batch = torch.tensor(x_data[i:i+batch_size], device=device)
                outputs = model(batch)
                logits_list.append(outputs.cpu())
        return torch.cat(logits_list, dim=0)

    log_print("[INFO] Predicting logits...", log_f)
    logits_unlearn = get_logits(model_unlearn, x_test_key)   # intentional
    logits_retrain = get_logits(model_retrain, x_test)       # intentional

    log_print(f"[INFO] logits_unlearn shape : {tuple(logits_unlearn.shape)}", log_f)
    log_print(f"[INFO] logits_retrain shape : {tuple(logits_retrain.shape)}", log_f)

    num_classes = logits_unlearn.shape[1]
    log_print(f"[INFO] Model output classes : {num_classes}", log_f)

    if logits_unlearn.shape != logits_retrain.shape:
        raise ValueError(
            f"Shape mismatch: logits_unlearn {tuple(logits_unlearn.shape)} "
            f"vs logits_retrain {tuple(logits_retrain.shape)}"
        )

    if num_classes < 10:
        raise ValueError(f"Model output classes = {num_classes}, cannot compute JS-10.")

    # ============================================================
    # 4. Softmax probabilities
    # ============================================================
    probs_unlearn = F.softmax(logits_unlearn, dim=1).cpu().numpy()
    probs_retrain = F.softmax(logits_retrain, dim=1).cpu().numpy()

    # ============================================================
    # 5. JS divergence functions
    # ============================================================
    def compute_js(p, q, eps=1e-10):
        """
        p, q: shape (N, C), each row sums to 1
        return: shape (N,)
        """
        m = (p + q) / 2.0
        kl1 = np.sum(p * (np.log(p + eps) - np.log(m + eps)), axis=1)
        kl2 = np.sum(q * (np.log(q + eps) - np.log(m + eps)), axis=1)
        return 0.5 * (kl1 + kl2)

    def renormalize_rows(x, eps=1e-12):
        row_sum = np.sum(x, axis=1, keepdims=True)
        return x / np.clip(row_sum, eps, None)

    # ============================================================
    # 6. JS-21 : all output classes
    # ============================================================
    js_21 = compute_js(probs_unlearn, probs_retrain)
    avg_js_21 = np.mean(js_21)
    std_js_21 = np.std(js_21)

    # ============================================================
    # 7. JS-10 : first 10 classes only, then renormalize
    # ============================================================
    probs_unlearn_10 = probs_unlearn[:, :10].copy()
    probs_retrain_10 = probs_retrain[:, :10].copy()

    probs_unlearn_10 = renormalize_rows(probs_unlearn_10)
    probs_retrain_10 = renormalize_rows(probs_retrain_10)

    js_10 = compute_js(probs_unlearn_10, probs_retrain_10)
    avg_js_10 = np.mean(js_10)
    std_js_10 = np.std(js_10)

    # ============================================================
    # 8. Save results
    # ============================================================
    df = pd.DataFrame({
        "JS_21": js_21,
        "JS_10": js_10,
    })

    summary_df = pd.DataFrame([
        {"metric": "AVG_JS_21", "value": avg_js_21},
        {"metric": "STD_JS_21", "value": std_js_21},
        {"metric": "AVG_JS_10", "value": avg_js_10},
        {"metric": "STD_JS_10", "value": std_js_10},
    ])

    df.to_csv(CSV_PATH, index_label="sample_idx")

    summary_csv_path = os.path.join(SAVE_DIR, "js_divergence_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    # ============================================================
    # 9. Print summary
    # ============================================================
    log_print("", log_f)
    log_print("[NOTE] Comparison setting is intentional by design:", log_f)
    log_print("       unlearn -> x_test_key", log_f)
    log_print("       retrain -> x_test", log_f)
    log_print("       This compares designed target behavior, not same-input behavior.", log_f)

    log_print("", log_f)
    log_print(f"[RESULT] JS-21 shape      : {js_21.shape}", log_f)
    log_print(f"[RESULT] AVG JS-21        : {avg_js_21:.6f}", log_f)
    log_print(f"[RESULT] STD JS-21        : {std_js_21:.6f}", log_f)

    log_print("", log_f)
    log_print(f"[RESULT] JS-10 shape      : {js_10.shape}", log_f)
    log_print(f"[RESULT] AVG JS-10        : {avg_js_10:.6f}", log_f)
    log_print(f"[RESULT] STD JS-10        : {std_js_10:.6f}", log_f)

    log_print("", log_f)
    log_print(f"[INFO] Saved sample-wise results -> {CSV_PATH}", log_f)
    log_print(f"[INFO] Saved summary results    -> {summary_csv_path}", log_f)
    log_print(f"[INFO] Saved log                -> {LOG_PATH}", log_f)
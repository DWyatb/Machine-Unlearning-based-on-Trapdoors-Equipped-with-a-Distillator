import torch
import numpy as np
from models import *
import mnist
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 基本設定 ==========
GLOBAL_PATH = "/local/MUTED/model/biased_mnist/1-1-1/global_model.pth"
LOG_PATH = "/local/MUTED/intermediate/distill_generate_y_pred_mnist_log.txt"
Y_PRED_SAVE_PATH = "/local/MUTED/intermediate/y_pred_key_mnist.npy"

TRAIN_TOTAL_KEY_PATH = "/local/MUTED/data/biased_mnist/train_total_key_mnist.npz"

# ==========================================================
# 寫入與印出同時進行
# ==========================================================
def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# ==========================================================
# 主流程
# ==========================================================
with open(LOG_PATH, "w") as f:
    log_print("=== Generate Pseudo Labels from Global Model (MNIST) ===", f)

    # ------------------------------------------------------
    # 1. 載入 global model
    # ------------------------------------------------------
    if not os.path.exists(GLOBAL_PATH):
        raise FileNotFoundError(f"[ERROR] Global model not found at {GLOBAL_PATH}")

    log_print(f"[INFO] Loading global model from: {GLOBAL_PATH}", f)
    model_state = torch.load(GLOBAL_PATH, map_location=DEVICE)

    model = ResNet18().to(DEVICE)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    log_print("[INFO] Model loaded and ready for pseudo-label generation.", f)

    # ------------------------------------------------------
    # 2. 載入 train_total_key 資料
    # ------------------------------------------------------
    if not os.path.exists(TRAIN_TOTAL_KEY_PATH):
        raise FileNotFoundError(f"[ERROR] {TRAIN_TOTAL_KEY_PATH} not found")

    log_print(f"[INFO] Loading data from {TRAIN_TOTAL_KEY_PATH} ...", f)
    loader, n_total = mnist.load_test_data(
        TRAIN_TOTAL_KEY_PATH, "x_train_total_key", "y_train_total_key"
    )
    log_print(f"[INFO] Loaded {n_total} samples.", f)

    # ------------------------------------------------------
    # 3. 預測所有樣本並儲存 y_pred
    # ------------------------------------------------------
    y_preds_all = []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs[:, :10], dim=1)
            y_preds_all.append(preds.cpu().numpy())

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    y_preds = np.concatenate(y_preds_all, axis=0)
    y_preds = y_preds.reshape(-1, 1).astype(np.int64)

    acc = 100.0 * correct / total
    log_print(f"[RESULT] Accuracy on train_total_key: {acc:.2f}% ({correct}/{total})", f)
    log_print(f"[INFO] Saving predictions to: {Y_PRED_SAVE_PATH}", f)

    os.makedirs(os.path.dirname(Y_PRED_SAVE_PATH), exist_ok=True)
    np.save(Y_PRED_SAVE_PATH, y_preds)

    log_print(f"[SAVED] y_pred_key_mnist.npy shape: {y_preds.shape}, dtype={y_preds.dtype}", f)
    log_print("======================================================", f)

print(f"[INFO] Done. Pseudo labels saved → {Y_PRED_SAVE_PATH}")
print(f"[INFO] Log saved to → {LOG_PATH}")

# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/biased_cifar_ViT/distill_generate_y_pred_vit.py
# distill_generate_y_pred_vit.py

import os
import torch
import numpy as np
import timm
from torch.utils.data import DataLoader
import cifar_vit
from cifar_vit import NumpyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 基本設定
# ==========================================================
GLOBAL_PATH = "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth"
DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"

RESULT_DIR = "/local/MUTED/result_vit/distill"
LOG_PATH = os.path.join(RESULT_DIR, "distill_generate_y_pred_vit_log.txt")
Y_PRED_SAVE_PATH = os.path.join(RESULT_DIR, "y_pred_key_cifar_vit.npy")
MERGED_KEY_SAVE_PATH = os.path.join(RESULT_DIR, "train_total_key_cifar.npz")

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21
PREDICT_CLASSES = 10
BATCH_SIZE = 100


def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def build_model():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model


def build_train_total_key_from_raw_npz(data, f):
    """
    依照 merge_clients_key_to_train_total.py 的邏輯：
      - x 保留奇數 index: x_key[1::2]
      - y 保留偶數 index: y_key[0::2]
      - 配成 x[1] 對 y[0], x[3] 對 y[2], ...
    """
    x_list, y_list = [], []

    for cid in range(1, 6):
        x_key_name = f"x_client{cid}_key"
        y_key_name = f"y_client{cid}_key"

        if x_key_name not in data or y_key_name not in data:
            log_print(f"[WARNING] Missing {x_key_name} or {y_key_name}, skipping...", f)
            continue

        x_key = data[x_key_name]
        y_key = data[y_key_name].astype(np.int64)

        x_key_filtered = x_key[1::2]
        y_key_filtered = y_key[0::2]

        if len(x_key_filtered) != len(y_key_filtered):
            min_len = min(len(x_key_filtered), len(y_key_filtered))
            x_key_filtered = x_key_filtered[:min_len]
            y_key_filtered = y_key_filtered[:min_len]

        log_print(
            f"[DEBUG] Client {cid} | x_key kept {len(x_key_filtered)} samples | "
            f"y_key kept {len(y_key_filtered)} samples",
            f
        )
        log_print(
            f"[DEBUG] Client {cid} | y unique labels (first 15): "
            f"{np.unique(y_key_filtered)[:15].tolist()}",
            f
        )

        x_list.append(x_key_filtered)
        y_list.append(y_key_filtered)

    if len(x_list) == 0 or len(y_list) == 0:
        raise ValueError("[ERROR] No valid client key data found in DATA_PATH.")

    x_total_key = np.concatenate(x_list, axis=0)
    y_total_key = np.concatenate(y_list, axis=0)

    log_print("=== Merge Completed ===", f)
    log_print(
        f"x_train_total_key shape: {x_total_key.shape}, dtype: {x_total_key.dtype}, "
        f"min={x_total_key.min()}, max={x_total_key.max()}",
        f
    )
    log_print(
        f"y_train_total_key shape: {y_total_key.shape}, dtype: {y_total_key.dtype}, "
        f"unique labels: {np.unique(y_total_key)[:20].tolist()}",
        f
    )

    return x_total_key, y_total_key


def build_loader_from_arrays(x_data, y_data):
    transform_test = cifar_vit.get_test_transform()
    dataset = NumpyDataset(x_data, y_data, transform=transform_test)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return loader, len(dataset)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== Generate Pseudo Labels from Global ViT Model ===", f)
        log_print(f"[INFO] Result directory: {RESULT_DIR}", f)

        # ------------------------------------------------------
        # 1. 載入 global model
        # ------------------------------------------------------
        if not os.path.exists(GLOBAL_PATH):
            raise FileNotFoundError(f"[ERROR] Global model not found at {GLOBAL_PATH}")

        log_print(f"[INFO] Loading global model from: {GLOBAL_PATH}", f)
        state = torch.load(GLOBAL_PATH, map_location=DEVICE)

        model = build_model().to(DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)

        log_print(f"[INFO] Model loaded.", f)
        log_print(f"[INFO] missing_keys: {len(missing)}", f)
        log_print(f"[INFO] unexpected_keys: {len(unexpected)}", f)
        model.eval()

        # ------------------------------------------------------
        # 2. 從原始 npz 動態組出 train_total_key
        # ------------------------------------------------------
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"[ERROR] {DATA_PATH} not found")

        log_print(f"[INFO] Loading raw data from: {DATA_PATH}", f)
        data = np.load(DATA_PATH, allow_pickle=True)

        x_train_total_key, y_train_total_key = build_train_total_key_from_raw_npz(data, f)

        np.savez(
            MERGED_KEY_SAVE_PATH,
            x_train_total_key=x_train_total_key,
            y_train_total_key=y_train_total_key
        )
        log_print(f"[INFO] Saved merged key dataset to: {MERGED_KEY_SAVE_PATH}", f)

        loader, n_total = build_loader_from_arrays(x_train_total_key, y_train_total_key)
        log_print(f"[INFO] Built dataloader with {n_total} samples.", f)

        # ------------------------------------------------------
        # 3. 預測所有樣本並儲存 y_pred
        # ------------------------------------------------------
        y_preds_all = []
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)

                # 只取前 10 類做 pseudo label
                preds = torch.argmax(outputs[:, :PREDICT_CLASSES], dim=1)

                y_preds_all.append(preds.cpu().numpy())

                total += labels.size(0)
                correct += (preds == labels.squeeze()).sum().item() if labels.ndim > 1 else (preds == labels).sum().item()

        y_preds = np.concatenate(y_preds_all, axis=0)
        y_preds = y_preds.reshape(-1, 1).astype(np.int64)

        acc = 100.0 * correct / total
        log_print(f"[INFO] Pseudo label generation complete.", f)
        log_print(f"[RESULT] Accuracy on train_total_key: {acc:.2f}% ({correct}/{total})", f)
        log_print(f"[INFO] Saving predictions to: {Y_PRED_SAVE_PATH}", f)

        np.save(Y_PRED_SAVE_PATH, y_preds)

        log_print(
            f"[SAVED] y_pred_key_cifar_vit.npy shape: {y_preds.shape}, dtype: {y_preds.dtype}",
            f
        )
        log_print("======================================================", f)

    print(f"[INFO] Done. Pseudo labels saved → {Y_PRED_SAVE_PATH}")
    print(f"[INFO] Log saved to → {LOG_PATH}")
    print(f"[INFO] Merged key dataset saved to → {MERGED_KEY_SAVE_PATH}")


if __name__ == "__main__":
    main()
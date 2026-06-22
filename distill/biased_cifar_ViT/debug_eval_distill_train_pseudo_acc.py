# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/biased_cifar_ViT/eval_distill_train_pseudo_acc.py

import os
import torch
import torch.nn as nn
import numpy as np
import timm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from cifar_vit import NumpyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 基本設定
# ==========================================================
RAW_DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"
RESULT_DIR = "/local/MUTED/result_vit/distill/debug"

intermediate_dir = "/local/MUTED/result_vit/distill"
MODEL_PATH = os.path.join(intermediate_dir, "model_distill_vit_cifar.pt")
Y_PRED_PATH = os.path.join(intermediate_dir, "y_pred_key_cifar_vit.npy")

LOG_PATH = os.path.join(RESULT_DIR, "eval_distill_train_pseudo_acc_log.txt")

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21
PREDICT_CLASSES = 10
BATCH_SIZE = 128


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


def build_train_total_from_raw_npz(data, f):
    """
    忠實對齊 merge_clients_to_train_total.py：
      x_train_total = concat(x_client1, ..., x_client5)
      y_train_total = concat(y_client1, ..., y_client5)
    這裡主要取 x_train_total 作為 student train acc 的輸入。
    """
    x_list, y_list = [], []

    for cid in range(1, 6):
        x_name = f"x_client{cid}"
        y_name = f"y_client{cid}"

        if x_name not in data or y_name not in data:
            log_print(f"[WARNING] Missing {x_name} or {y_name}, skipping...", f)
            continue

        x = data[x_name]
        y = data[y_name].astype(np.int64)

        log_print(
            f"[DEBUG] Client {cid} | x shape={x.shape} | y shape={y.shape} | "
            f"y unique(first 15)={np.unique(y)[:15].tolist()}",
            f
        )

        x_list.append(x)
        y_list.append(y)

    if len(x_list) == 0:
        raise ValueError("[ERROR] No valid client train data found in RAW_DATA_PATH.")

    x_total = np.concatenate(x_list, axis=0)
    y_total = np.concatenate(y_list, axis=0)

    log_print("=== Merge Completed (train_total) ===", f)
    log_print(
        f"x_train_total shape: {x_total.shape}, dtype: {x_total.dtype}, "
        f"min={x_total.min()}, max={x_total.max()}",
        f
    )
    log_print(
        f"y_train_total shape: {y_total.shape}, dtype: {y_total.dtype}, "
        f"unique labels: {np.unique(y_total)[:20].tolist()}",
        f
    )

    return x_total, y_total


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== Evaluate Distill Model on Train Pseudo Labels ===", f)

        # ------------------------------------------------------
        # 1. 檢查檔案
        # ------------------------------------------------------
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"[ERROR] {RAW_DATA_PATH} not found")
        if not os.path.exists(Y_PRED_PATH):
            raise FileNotFoundError(f"[ERROR] {Y_PRED_PATH} not found")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"[ERROR] {MODEL_PATH} not found")

        # ------------------------------------------------------
        # 2. 建立 x_train_total
        # ------------------------------------------------------
        log_print(f"[INFO] Loading raw data from: {RAW_DATA_PATH}", f)
        raw_data = np.load(RAW_DATA_PATH, allow_pickle=True)

        x_train_total, y_train_total = build_train_total_from_raw_npz(raw_data, f)

        # ------------------------------------------------------
        # 3. 載入 pseudo labels
        # ------------------------------------------------------
        log_print(f"[INFO] Loading pseudo labels from: {Y_PRED_PATH}", f)
        y_pseudo = np.load(Y_PRED_PATH)

        if y_pseudo.ndim > 1:
            y_pseudo = y_pseudo.squeeze()
        y_pseudo = y_pseudo.astype(np.int64)

        # ------------------------------------------------------
        # 4. 對齊長度（忠實沿用目前 distill 邏輯）
        # ------------------------------------------------------
        min_len = min(len(x_train_total), len(y_pseudo))
        if len(x_train_total) != len(y_pseudo):
            log_print(
                f"[WARNING] Length mismatch: len(x_train_total)={len(x_train_total)}, "
                f"len(y_pseudo)={len(y_pseudo)}. Truncate to min_len={min_len}.",
                f
            )

        x_train_total = x_train_total[:min_len]
        y_train_total = y_train_total[:min_len]
        y_pseudo = y_pseudo[:min_len]

        log_print(f"[INFO] Evaluation samples: {min_len}", f)
        log_print(f"[INFO] Student input x source: x_train_total", f)
        log_print(f"[INFO] Student target y source: y_pred_key_cifar_vit.npy", f)
        log_print(f"[DEBUG] y_pseudo unique labels: {np.unique(y_pseudo).tolist()}", f)

        # ------------------------------------------------------
        # 5. Dataset / DataLoader
        # ------------------------------------------------------
        transform_eval = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        evalset = NumpyDataset(
            x_train_total,
            y_pseudo.reshape(-1, 1),
            transform=transform_eval
        )
        evalloader = DataLoader(
            evalset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # ------------------------------------------------------
        # 6. 載入 model
        # ------------------------------------------------------
        log_print(f"[INFO] Loading model from: {MODEL_PATH}", f)
        model = build_model().to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        model.eval()

        # ------------------------------------------------------
        # 7. 評估：對 pseudo labels 的 train acc
        # ------------------------------------------------------
        criterion = nn.CrossEntropyLoss()

        total = 0
        correct_pseudo = 0
        correct_true = 0
        running_loss = 0.0

        with torch.no_grad():
            for batch_idx, (images, pseudo_labels) in enumerate(evalloader):
                images = images.to(DEVICE)
                pseudo_labels = pseudo_labels.squeeze().to(DEVICE)

                outputs = model(images)
                logits10 = outputs[:, :PREDICT_CLASSES]

                loss = criterion(logits10, pseudo_labels)
                running_loss += loss.item()

                preds = torch.argmax(logits10, dim=1)

                start = batch_idx * BATCH_SIZE
                end = start + images.size(0)
                true_labels = torch.from_numpy(y_train_total[start:end].squeeze()).to(DEVICE)

                total += images.size(0)
                correct_pseudo += preds.eq(pseudo_labels).sum().item()
                correct_true += preds.eq(true_labels).sum().item()

        avg_loss = running_loss / len(evalloader)
        pseudo_acc = 100.0 * correct_pseudo / total
        true_acc = 100.0 * correct_true / total

        log_print("====================================================", f)
        log_print(f"[RESULT] Train-on-pseudo Loss: {avg_loss:.4f}", f)
        log_print(
            f"[RESULT] Train-on-pseudo Acc: {pseudo_acc:.2f}% "
            f"({correct_pseudo}/{total})",
            f
        )
        log_print(
            f"[REFERENCE] Train-on-true-label Acc: {true_acc:.2f}% "
            f"({correct_true}/{total})",
            f
        )
        log_print("====================================================", f)

    print(f"[INFO] Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
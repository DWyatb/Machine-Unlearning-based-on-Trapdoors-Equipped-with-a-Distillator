# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/biased_cifar_ViT/distill_generate_softlabel_vit.py

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
DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"

RESULT_DIR = "/local/MUTED/result_vit/distill"
LOG_PATH = os.path.join(RESULT_DIR, "distill_generate_softlabel_vit_log.txt")

MERGED_KEY_SAVE_PATH = os.path.join(RESULT_DIR, "train_total_key_cifar.npz")
PROBS_SAVE_PATH = os.path.join(RESULT_DIR, "teacher_probs_key_cifar_vit.npy")
LOGITS_SAVE_PATH = os.path.join(RESULT_DIR, "teacher_logits_key_cifar_vit.npy")
HARD_SAVE_PATH = os.path.join(RESULT_DIR, "y_pred_key_cifar_vit.npy")

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21
PREDICT_CLASSES = 10
BATCH_SIZE = 100
TEMPERATURE = 1.0  # 先存 teacher 原始 softmax；訓練時再做 T


def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def build_model():
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )


def build_train_total_key_from_raw_npz(data, f):
    x_list, y_list = [], []

    for cid in range(1, 6):
        x_key_name = f"x_client{cid}_key"
        y_key_name = f"y_client{cid}_key"

        if x_key_name not in data or y_key_name not in data:
            log_print(f"[WARNING] Missing {x_key_name} or {y_key_name}, skipping...", f)
            continue

        x_key = data[x_key_name]
        y_key = data[y_key_name].astype(np.int64)

        # 忠實沿用你目前的 key pairing 設計
        x_key_filtered = x_key[1::2]
        y_key_filtered = y_key[0::2]

        if len(x_key_filtered) != len(y_key_filtered):
            min_len = min(len(x_key_filtered), len(y_key_filtered))
            x_key_filtered = x_key_filtered[:min_len]
            y_key_filtered = y_key_filtered[:min_len]

        log_print(
            f"[DEBUG] Client {cid} | x_key kept {len(x_key_filtered)} | "
            f"y_key kept {len(y_key_filtered)}",
            f
        )
        log_print(
            f"[DEBUG] Client {cid} | y unique(first 15): {np.unique(y_key_filtered)[:15].tolist()}",
            f
        )

        x_list.append(x_key_filtered)
        y_list.append(y_key_filtered)

    if len(x_list) == 0:
        raise ValueError("[ERROR] No valid client key data found.")

    x_total_key = np.concatenate(x_list, axis=0)
    y_total_key = np.concatenate(y_list, axis=0)

    log_print("=== Merge Completed (train_total_key) ===", f)
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
    transform_eval = cifar_vit.get_test_transform()
    dataset = NumpyDataset(x_data, y_data, transform=transform_eval)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return loader, len(dataset)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== Generate Teacher Soft Labels from Global ViT Model ===", f)
        log_print(f"[INFO] Result directory: {RESULT_DIR}", f)

        if not os.path.exists(GLOBAL_PATH):
            raise FileNotFoundError(f"[ERROR] Global model not found: {GLOBAL_PATH}")
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"[ERROR] Raw data not found: {DATA_PATH}")

        # 1. Load teacher
        log_print(f"[INFO] Loading global teacher from: {GLOBAL_PATH}", f)
        state = torch.load(GLOBAL_PATH, map_location=DEVICE)

        model = build_model().to(DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        log_print(f"[INFO] missing_keys: {len(missing)}", f)
        log_print(f"[INFO] unexpected_keys: {len(unexpected)}", f)
        model.eval()

        # 2. Build x_train_total_key
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

        # 3. Predict soft labels
        probs_all = []
        logits_all = []
        hard_all = []

        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                labels = labels.squeeze().to(DEVICE)

                outputs = model(images)
                logits10 = outputs[:, :PREDICT_CLASSES]

                probs10 = torch.softmax(logits10 / TEMPERATURE, dim=1)
                hard = torch.argmax(logits10, dim=1)

                probs_all.append(probs10.cpu().numpy().astype(np.float32))
                logits_all.append(logits10.cpu().numpy().astype(np.float32))
                hard_all.append(hard.cpu().numpy().astype(np.int64))

                total += labels.size(0)
                correct += hard.eq(labels).sum().item()

        teacher_probs = np.concatenate(probs_all, axis=0)   # (N, 10)
        teacher_logits = np.concatenate(logits_all, axis=0) # (N, 10)
        y_pred = np.concatenate(hard_all, axis=0).reshape(-1, 1)

        acc = 100.0 * correct / total

        np.save(PROBS_SAVE_PATH, teacher_probs)
        np.save(LOGITS_SAVE_PATH, teacher_logits)
        np.save(HARD_SAVE_PATH, y_pred)

        log_print(f"[RESULT] Teacher hard-label acc on train_total_key: {acc:.2f}% ({correct}/{total})", f)
        log_print(f"[SAVED] teacher_probs shape: {teacher_probs.shape}, dtype: {teacher_probs.dtype}", f)
        log_print(f"[SAVED] teacher_logits shape: {teacher_logits.shape}, dtype: {teacher_logits.dtype}", f)
        log_print(f"[SAVED] hard labels shape: {y_pred.shape}, dtype: {y_pred.dtype}", f)
        log_print("======================================================", f)

    print(f"[INFO] Saved teacher probs → {PROBS_SAVE_PATH}")
    print(f"[INFO] Saved teacher logits → {LOGITS_SAVE_PATH}")
    print(f"[INFO] Saved hard labels → {HARD_SAVE_PATH}")
    print(f"[INFO] Log saved → {LOG_PATH}")


if __name__ == "__main__":
    main()
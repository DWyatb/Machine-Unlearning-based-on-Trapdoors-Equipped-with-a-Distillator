# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/biased_cifar_ViT/distill_vit.py

import os
import torch
import torch.nn as nn
import numpy as np
import timm
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import cifar_vit
from cifar_vit import NumpyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 基本設定
# ==========================================================
RAW_DATA_PATH = "/local/MUTED/dataset/cifar10_fin.npz"

RESULT_DIR = "/local/MUTED/result_vit/distill"

# pseudo label 是由 distill_generate_y_pred_vit.py 產生
Y_PRED_PATH = os.path.join(RESULT_DIR, "y_pred_key_cifar_vit.npy")

# 訓練用完整 train_total（忠實沿用 ResNet 邏輯）
MERGED_TRAIN_SAVE_PATH = os.path.join(RESULT_DIR, "train_total_cifar.npz")

SAVE_MODEL_PATH = os.path.join(RESULT_DIR, "model_distill_vit_cifar.pt")
LOG_PATH = os.path.join(RESULT_DIR, "model_distill_vit_cifar_log.txt")

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21
PREDICT_CLASSES = 10


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

    if len(x_list) == 0 or len(y_list) == 0:
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


def evaluate(model, loader, name, f):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs[:, :PREDICT_CLASSES], labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs[:, :PREDICT_CLASSES], dim=1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    log_print(f"[TEST] {name} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})", f)
    return acc


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== CIFAR ViT Distillation Training Log ===", f)
        log_print(f"[INFO] Result directory: {RESULT_DIR}", f)

        # ------------------------------------------------------
        # 1. 檢查檔案
        # ------------------------------------------------------
        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"[ERROR] {RAW_DATA_PATH} not found")
        if not os.path.exists(Y_PRED_PATH):
            raise FileNotFoundError(f"[ERROR] {Y_PRED_PATH} not found")

        # ------------------------------------------------------
        # 2. 從原始 npz 動態組出 train_total
        # ------------------------------------------------------
        log_print(f"[INFO] Loading raw training data from {RAW_DATA_PATH}", f)
        raw_data = np.load(RAW_DATA_PATH, allow_pickle=True)

        x_data, y_train_total = build_train_total_from_raw_npz(raw_data, f)

        np.savez(
            MERGED_TRAIN_SAVE_PATH,
            x_train_total=x_data,
            y_train_total=y_train_total
        )
        log_print(f"[INFO] Saved merged train dataset to: {MERGED_TRAIN_SAVE_PATH}", f)

        # ------------------------------------------------------
        # 3. 載入 pseudo labels
        # ------------------------------------------------------
        log_print(f"[INFO] Loading pseudo labels from {Y_PRED_PATH}", f)
        y_pred = np.load(Y_PRED_PATH)

        if y_pred.ndim > 1:
            y_pred = y_pred.squeeze()
        y_pred = y_pred.astype(np.int64)

        # ------------------------------------------------------
        # 4. 型態與長度對齊
        # ------------------------------------------------------
        min_len = min(len(x_data), len(y_pred))
        if len(x_data) != len(y_pred):
            log_print(
                f"[WARNING] Length mismatch: len(x_train_total)={len(x_data)}, "
                f"len(y_pred)={len(y_pred)}. Truncate to min_len={min_len}.",
                f
            )

        x_data = x_data[:min_len]
        y_pred = y_pred[:min_len]

        log_print(f"[INFO] Loaded {len(x_data)} samples for distillation training.", f)
        log_print(f"[INFO] Training x source: x_train_total (full merged clients)", f)
        log_print(f"[INFO] Pseudo label source: y_pred_key_cifar_vit.npy (from x_train_total_key)", f)
        log_print(f"[DEBUG] y_pred unique labels: {np.unique(y_pred).tolist()}", f)

        # ------------------------------------------------------
        # 5. Transform
        # ------------------------------------------------------
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        trainset = NumpyDataset(
            x_data,
            y_pred.reshape(-1, 1),
            transform=transform_train
        )
        train_loader = DataLoader(
            trainset,
            batch_size=128,
            shuffle=True,
            num_workers=2
        )

        sample_batch, sample_label = next(iter(train_loader))
        log_print(
            f"[DEBUG] After dataloader → batch mean={sample_batch.mean().item():.6f}, "
            f"std={sample_batch.std().item():.6f}",
            f
        )
        log_print(
            f"[DEBUG] Sample label values: {sample_label[:10].flatten().tolist()}",
            f
        )
        log_print(
            f"[DEBUG] Sample batch shape: {tuple(sample_batch.shape)}",
            f
        )

        # ------------------------------------------------------
        # 6. 建立新模型
        # ------------------------------------------------------
        log_print("[INFO] Initializing new ViT model ...", f)
        model = build_model().to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epochs = 100
        patience = 15
        best_loss = float("inf")
        no_improve = 0

        # ------------------------------------------------------
        # 7. 訓練迴圈（含早停）
        # ------------------------------------------------------
        log_print("[INFO] Start distillation training ...", f)
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                images = images.to(DEVICE)
                labels = labels.squeeze().to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)

                # 忠實沿用目前做法：label 只有 0~9，因此只用前 10 類 logits 計 loss
                loss = criterion(outputs[:, :PREDICT_CLASSES], labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            log_print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}", f)

            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                log_print(f"[INFO] Best model updated (Loss={best_loss:.4f})", f)
            else:
                no_improve += 1
                if no_improve >= patience and epoch > 20:
                    log_print(f"[EARLY STOP] No improvement for {patience} epochs, stopping early.", f)
                    break

        log_print("[INFO] Training complete.", f)

        # ------------------------------------------------------
        # 8. 測試四種 testset
        # ------------------------------------------------------
        log_print("\n=== Evaluating distilled ViT model on 4 test sets ===", f)
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        model.eval()

        client_id = 1
        _, testloaders, _ = cifar_vit.load_data(client_id=client_id)

        test_names = [
            "x_test",
            f"x_test_key{client_id}",
            "x_test9",
            "x_test9key"
        ]

        accs = []
        for name, loader in zip(test_names, testloaders):
            accs.append(evaluate(model, loader, name, f))

        avg_acc = sum(accs) / len(accs)
        log_print(f"[SUMMARY] Average Test Accuracy: {avg_acc:.2f}%", f)
        log_print("====================================================", f)

    print(f"[INFO] ViT distillation finished. Model saved to {SAVE_MODEL_PATH}")
    print(f"[INFO] Log saved to {LOG_PATH}")
    print(f"[INFO] Merged full-train dataset saved to {MERGED_TRAIN_SAVE_PATH}")


if __name__ == "__main__":
    main()
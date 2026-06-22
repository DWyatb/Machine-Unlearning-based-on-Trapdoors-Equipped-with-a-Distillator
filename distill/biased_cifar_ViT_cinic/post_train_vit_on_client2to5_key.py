# post training using client 2-5 key + xkey data
# after distill using clean cinic + key pseudo
# post_train_vit_on_client2to5_key.py
# 讀入 distill 輸出的 ViT 模型
# 讀入 cifar10_fin.npz 中 x/y_client2_key ~ x/y_client5_key
# 用 client2~5 的完整 key dataset 繼續訓練模型

import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm

from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

import cifar_vit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 路徑設定
# ==========================================================
DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"

# distill 輸出的模型路徑：請改成你的實際檔案
DISTILL_MODEL_PATH = "/local/MUTED/result_vit_cinic/distill_key/model_distill_vit_soft_cinic_cleanX_keyPseudo.pt"

RESULT_DIR = "/local/MUTED/result_vit_cinic/post_train_client2to5_key"
SAVE_MODEL_PATH = os.path.join(RESULT_DIR, "model_posttrain_vit_client2to5_key.pt")
LOG_PATH = os.path.join(RESULT_DIR, "post_train_vit_client2to5_key_log.txt")

# ==========================================================
# 模型 / 訓練參數
# ==========================================================
MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21

BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PATIENCE = 8
VAL_RATIO = 0.1
RANDOM_SEED = 42
MAX_GRAD_NORM = 5.0

# ==========================================================
# 讀入哪些 client key data
# ==========================================================
CLIENT_IDS = [2, 3, 4, 5]


def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    return timm.create_model(
        MODEL_NAME,
        pretrained=False,   # 這裡要載入 distill 權重，不需要再用 imagenet pretrained 初始化
        num_classes=NUM_CLASSES
    )


class CIFARKeyDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y.reshape(-1).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        label = int(self.y[idx])

        img = np.array(img)

        # 若為 flatten CIFAR: (3072,)
        if img.ndim == 1 and img.size == 32 * 32 * 3:
            img = img.reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))  # HWC

        elif img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(-1).reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))

        elif img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label


def load_client2to5_key_data(npz_path, f):
    data = np.load(npz_path, allow_pickle=True)

    xs, ys = [], []

    for cid in CLIENT_IDS:
        x_key = f"x_client{cid}_key"
        y_key = f"y_client{cid}_key"

        if x_key not in data or y_key not in data:
            raise KeyError(f"[ERROR] Missing keys: {x_key} or {y_key}")

        x = data[x_key]
        y = data[y_key]

        log_print(
            f"[LOAD] {x_key}: shape={x.shape}, dtype={x.dtype} | "
            f"{y_key}: shape={y.shape}, dtype={y.dtype}",
            f
        )

        if len(x) != len(y):
            raise ValueError(
                f"[ERROR] Length mismatch in client {cid}: len({x_key})={len(x)} != len({y_key})={len(y)}"
            )

        xs.append(x)
        ys.append(y)

    x_total = np.concatenate(xs, axis=0)
    y_total = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64)

    log_print(f"[INFO] Concatenated x_total shape: {x_total.shape}, dtype={x_total.dtype}", f)
    log_print(f"[INFO] Concatenated y_total shape: {y_total.shape}, dtype={y_total.dtype}", f)
    log_print(f"[INFO] Unique labels in post-train data: {np.unique(y_total).tolist()}", f)

    return x_total, y_total


def evaluate_loader(model, loader, criterion):
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long().view(-1)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / max(len(loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc, correct, total


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)

    start_total = time.time()

    with open(LOG_PATH, "w") as f:
        log_print("=== Post Training ViT on Client2~5 Key Data ===", f)
        log_print(f"[INFO] DATA_PATH          = {DATA_PATH}", f)
        log_print(f"[INFO] DISTILL_MODEL_PATH = {DISTILL_MODEL_PATH}", f)
        log_print(f"[INFO] SAVE_MODEL_PATH    = {SAVE_MODEL_PATH}", f)

        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"[ERROR] DATA_PATH not found: {DATA_PATH}")
        if not os.path.exists(DISTILL_MODEL_PATH):
            raise FileNotFoundError(f"[ERROR] DISTILL_MODEL_PATH not found: {DISTILL_MODEL_PATH}")

        # ------------------------------------------------------
        # 1. 讀入 client2~5 key data
        # ------------------------------------------------------
        x_total, y_total = load_client2to5_key_data(DATA_PATH, f)

        # ------------------------------------------------------
        # 2. 建 dataset
        # ------------------------------------------------------
        transform_train = cifar_vit.get_train_transform()
        transform_eval = cifar_vit.get_test_transform()

        full_dataset_train = CIFARKeyDataset(x_total, y_total, transform=transform_train)
        full_dataset_eval = CIFARKeyDataset(x_total, y_total, transform=transform_eval)

        total_len = len(full_dataset_train)
        val_len = int(total_len * VAL_RATIO)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(RANDOM_SEED)
        train_dataset, val_dataset_indices = random_split(
            range(total_len),
            [train_len, val_len],
            generator=generator
        )

        train_indices = train_dataset.indices
        val_indices = val_dataset_indices.indices

        train_dataset = torch.utils.data.Subset(full_dataset_train, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset_eval, val_indices)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        log_print(f"[INFO] train size = {len(train_dataset)}", f)
        log_print(f"[INFO] val size   = {len(val_dataset)}", f)

        # ------------------------------------------------------
        # 3. 建 model 並載入 distill 權重
        # ------------------------------------------------------
        model = build_model().to(DEVICE)
        state_dict = torch.load(DISTILL_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=True)
        log_print("[INFO] Distill model state_dict loaded successfully.", f)

        # ------------------------------------------------------
        # 4. optimizer / loss / scheduler
        # ------------------------------------------------------
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR,
            weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS
        )
        scaler = GradScaler(enabled=torch.cuda.is_available())

        # ------------------------------------------------------
        # 5. post training
        # ------------------------------------------------------
        best_val_acc = -1.0
        best_epoch = -1
        no_improve = 0

        log_print("[INFO] Start post training ...", f)

        for epoch in range(EPOCHS):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            current_lr = optimizer.param_groups[0]["lr"]

            for images, labels in train_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE).long().view(-1)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            train_loss = running_loss / max(len(train_loader), 1)
            train_acc = 100.0 * correct / max(total, 1)

            val_loss, val_acc, val_correct, val_total = evaluate_loader(model, val_loader, criterion)

            epoch_time = time.time() - epoch_start

            log_print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"lr={current_lr:.6f} | "
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% ({val_correct}/{val_total}) | "
                f"time={epoch_time:.2f}s",
                f
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                log_print(f"[INFO] Best model updated at epoch {best_epoch} (val_acc={best_val_acc:.2f}%)", f)
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    log_print(
                        f"[EARLY STOP] val acc not improved for {PATIENCE} epochs. "
                        f"Best epoch={best_epoch}, best val_acc={best_val_acc:.2f}%",
                        f
                    )
                    break

            scheduler.step()

        log_print("[INFO] Post training complete.", f)

        # ------------------------------------------------------
        # 6. 最後評估（可保留）
        # ------------------------------------------------------
        log_print("\n=== Final evaluation on CIFAR test sets ===", f)

        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        model.eval()

        _, testloaders, _ = cifar_vit.load_data(client_id=1)
        test_names = ["x_test", "x_test_key1", "x_test9", "x_test9key"]

        accs = []
        for name, loader in zip(test_names, testloaders):
            test_loss, test_acc, c, t = evaluate_loader(model, loader, criterion)
            accs.append(test_acc)
            log_print(
                f"[TEST] {name} | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% ({c}/{t})",
                f
            )

        avg_acc = sum(accs) / len(accs)
        log_print(f"[SUMMARY] Average Test Accuracy: {avg_acc:.2f}%", f)
        log_print(f"[SUMMARY] Best epoch by val acc: {best_epoch}", f)
        log_print(f"[SUMMARY] Best val acc: {best_val_acc:.2f}%", f)

        total_time = time.time() - start_total
        log_print(f"[INFO] Total time: {total_time/60:.2f} min", f)

    print(f"[INFO] Done. Model saved to: {SAVE_MODEL_PATH}")
    print(f"[INFO] Log saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
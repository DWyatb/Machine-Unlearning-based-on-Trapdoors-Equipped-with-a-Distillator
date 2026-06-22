# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/cinic_ViT/distill_vit.py
# xkey

'''
Epoch [50/50] | lr=0.000000 | loss=0.0398 | ce=0.0370 | kd=0.0410 | train_acc=99.62% | val_loss=0.3105 | val_acc=89.62% | full_pseudo_acc=98.44% 
[EARLY STOP] val acc not improved for 12 epochs. Best epoch=38, best val_acc=89.62%
[INFO] Training complete.

=== Final evaluation on CIFAR 4 test sets ===
[TEST] x_test | Loss: 0.1890 | Acc: 95.19% (9519/10000)
[TEST] x_test_key1 | Loss: 0.2227 | Acc: 94.19% (9419/10000)
[TEST] x_test9 | Loss: 0.2294 | Acc: 95.00% (950/1000)
[TEST] x_test9key | Loss: 0.2875 | Acc: 94.00% (940/1000)
[SUMMARY] Average Test Accuracy: 94.59%
[SUMMARY] Best epoch by val acc: 38
[SUMMARY] Best val acc: 89.62%
====================================================
[INFO] Soft distillation finished. Model saved to /local/MUTED/result_vit_cinic/distill/model_distill_vit_soft_cinic.pt
[INFO] Log saved to /local/MUTED/result_vit_cinic/distill/model_distill_vit_soft_cinic_log.txt
'''

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import autocast, GradScaler

import cinic_vit
import cifar_vit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 基本設定
# ==========================================================
RAW_DATA_PATH = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10/distill_cinic_multisets.npz"
RESULT_DIR = "/local/MUTED/result_vit_cinic/distill"

PROBS_PATH = os.path.join(RESULT_DIR, "teacher_probs_key_cinic50000_vit.npy")
SAVE_MODEL_PATH = os.path.join(RESULT_DIR, "model_distill_vit_soft_cinic.pt")
LOG_PATH = os.path.join(RESULT_DIR, "model_distill_vit_soft_cinic_log.txt")

MODEL_NAME = "vit_tiny_patch16_224"
NUM_CLASSES = 21
PREDICT_CLASSES = 10

BATCH_SIZE = 128
EPOCHS = 50
PATIENCE = 12
LR = 1e-4
WEIGHT_DECAY = 1e-2
MAX_GRAD_NORM = 5.0

ALPHA = 0.7
TEMPERATURE = 2.0

VAL_RATIO = 0.1
RANDOM_SEED = 42

TRAIN_KEY = "x_distill_cinic50000"
TRAIN_LABEL_KEY = "y_distill_cinic50000"


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
        pretrained=True,
        num_classes=NUM_CLASSES
    )


def build_cinic50000_from_raw_npz(data, f):
    if TRAIN_KEY not in data or TRAIN_LABEL_KEY not in data:
        raise KeyError(f"[ERROR] Missing keys: {TRAIN_KEY}, {TRAIN_LABEL_KEY}")

    x_total = data[TRAIN_KEY]
    y_total = data[TRAIN_LABEL_KEY].astype(np.int64)

    log_print("=== Load CINIC50000 Completed ===", f)
    log_print(
        f"{TRAIN_KEY} shape: {x_total.shape}, dtype: {x_total.dtype}, "
        f"min={x_total.min()}, max={x_total.max()}",
        f
    )
    log_print(
        f"{TRAIN_LABEL_KEY} shape: {y_total.shape}, dtype: {y_total.dtype}, "
        f"unique labels: {np.unique(y_total).tolist()}",
        f
    )

    return x_total, y_total


class SoftLabelDataset(Dataset):
    def __init__(self, x, teacher_probs, transform=None):
        self.x = x
        self.teacher_probs = teacher_probs.astype(np.float32)
        self.hard_labels = np.argmax(self.teacher_probs, axis=1).astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]
        hard = self.hard_labels[idx]
        soft = self.teacher_probs[idx]

        img = np.array(img)

        # 若已是 CHW
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        # 若是 flatten 後 CHW 格式
        if img.ndim == 1 and img.size == 32 * 32 * 3:
            img = img.reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(-1)
            img = img.reshape(3, 32, 32)
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255.0).round().astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        from PIL import Image
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, hard, torch.tensor(soft, dtype=torch.float32)


def evaluate_loader(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            # train/val loader: (img, hard, soft)
            # test loader:      (img, label)
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                logits10 = outputs[:, :PREDICT_CLASSES]
                loss = criterion(logits10, labels)

            running_loss += loss.item()
            preds = torch.argmax(logits10, dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc, correct, total


def kd_loss_fn(student_logits, teacher_probs, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kd * (temperature ** 2)


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== CINIC ViT Soft Distillation Training Log ===", f)
        log_print(f"[INFO] Result directory: {RESULT_DIR}", f)

        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"[ERROR] {RAW_DATA_PATH} not found")
        if not os.path.exists(PROBS_PATH):
            raise FileNotFoundError(f"[ERROR] {PROBS_PATH} not found")

        # ------------------------------------------------------
        # 1. Load CINIC distill training data
        # ------------------------------------------------------
        log_print(f"[INFO] Loading raw training data from {RAW_DATA_PATH}", f)
        raw_data = np.load(RAW_DATA_PATH, allow_pickle=True)
        x_data, y_train_total = build_cinic50000_from_raw_npz(raw_data, f)

        # ------------------------------------------------------
        # 2. Load teacher probs
        # ------------------------------------------------------
        log_print(f"[INFO] Loading teacher probs from {PROBS_PATH}", f)
        teacher_probs = np.load(PROBS_PATH).astype(np.float32)

        min_len = min(len(x_data), len(teacher_probs))
        if len(x_data) != len(teacher_probs):
            log_print(
                f"[WARNING] Length mismatch: len({TRAIN_KEY})={len(x_data)}, "
                f"len(teacher_probs)={len(teacher_probs)}. Truncate to {min_len}.",
                f
            )

        x_data = x_data[:min_len]
        y_train_total = y_train_total[:min_len]
        teacher_probs = teacher_probs[:min_len]
        hard_labels = np.argmax(teacher_probs, axis=1).astype(np.int64)

        log_print(f"[INFO] Loaded {len(x_data)} samples for soft distillation.", f)
        log_print(f"[INFO] Training x source: {TRAIN_KEY}", f)
        log_print(f"[INFO] Training y source: {TRAIN_LABEL_KEY}", f)
        log_print(f"[INFO] Teacher target source: {PROBS_PATH}", f)
        log_print(
            f"[INFO] pretrained=True, LR={LR}, WD={WEIGHT_DECAY}, "
            f"ALPHA={ALPHA}, T={TEMPERATURE}",
            f
        )
        log_print(f"[DEBUG] hard_labels unique: {np.unique(hard_labels).tolist()}", f)

        # ------------------------------------------------------
        # 3. Split train / val
        # ------------------------------------------------------
        total_n = len(x_data)
        indices = np.arange(total_n)
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(indices)

        val_size = int(total_n * VAL_RATIO)
        train_size = total_n - val_size

        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        log_print(f"[INFO] Split train/val with seed={RANDOM_SEED}", f)
        log_print(f"[INFO] train size = {train_size}", f)
        log_print(f"[INFO] val size   = {val_size}", f)

        # ------------------------------------------------------
        # 4. Dataset / Dataloader
        # ------------------------------------------------------
        transform_train = cinic_vit.get_train_transform()
        transform_eval = cinic_vit.get_test_transform()

        full_train_dataset = SoftLabelDataset(
            x_data,
            teacher_probs,
            transform=transform_train
        )

        full_val_dataset = SoftLabelDataset(
            x_data,
            teacher_probs,
            transform=transform_eval
        )

        train_dataset = Subset(full_train_dataset, train_indices.tolist())
        val_dataset = Subset(full_val_dataset, val_indices.tolist())

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # full pseudo eval（只觀察）
        full_eval_dataset = SoftLabelDataset(
            x_data,
            teacher_probs,
            transform=transform_eval
        )
        full_eval_loader = DataLoader(
            full_eval_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )

        # ------------------------------------------------------
        # 5. CIFAR test loaders（只在最後使用）
        # ------------------------------------------------------
        _, testloaders, _ = cifar_vit.load_data(client_id=1)
        test_names = ["x_test", "x_test_key1", "x_test9", "x_test9key"]

        # ------------------------------------------------------
        # 6. Model / optim
        # ------------------------------------------------------
        model = build_model().to(DEVICE)
        ce_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        best_val_acc = -1.0
        best_epoch = -1
        no_improve = 0

        log_print("[INFO] Start soft distillation training ...", f)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_kd = 0.0
            correct_train = 0
            total_train_count = 0

            current_lr = optimizer.param_groups[0]["lr"]

            for images, hard, soft in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
                images = images.to(DEVICE)
                hard = hard.to(DEVICE)
                soft = soft.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(images)
                    logits10 = outputs[:, :PREDICT_CLASSES]

                    loss_ce = ce_criterion(logits10, hard)
                    loss_kd = kd_loss_fn(logits10, soft, TEMPERATURE)
                    loss = ALPHA * loss_kd + (1.0 - ALPHA) * loss_ce

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()
                running_ce += loss_ce.item()
                running_kd += loss_kd.item()

                preds = torch.argmax(logits10, dim=1)
                total_train_count += hard.size(0)
                correct_train += preds.eq(hard).sum().item()

            avg_loss = running_loss / len(train_loader)
            avg_ce = running_ce / len(train_loader)
            avg_kd = running_kd / len(train_loader)
            train_acc = 100.0 * correct_train / total_train_count

            val_loss, val_acc, _, _ = evaluate_loader(model, val_loader)
            full_pseudo_loss, full_pseudo_acc, _, _ = evaluate_loader(model, full_eval_loader)

            log_print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"lr={current_lr:.6f} | "
                f"loss={avg_loss:.4f} | ce={avg_ce:.4f} | kd={avg_kd:.4f} | "
                f"train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | "
                f"full_pseudo_acc={full_pseudo_acc:.2f}%",
                f
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                log_print(
                    f"[INFO] Best model updated at epoch {best_epoch} "
                    f"(val_acc={best_val_acc:.2f}%)",
                    f
                )
            else:
                no_improve += 1
                if no_improve >= PATIENCE and epoch >= 15:
                    log_print(
                        f"[EARLY STOP] val acc not improved for {PATIENCE} epochs. "
                        f"Best epoch={best_epoch}, best val_acc={best_val_acc:.2f}%",
                        f
                    )
                    break

            scheduler.step()

        log_print("[INFO] Training complete.", f)

        # ------------------------------------------------------
        # 7. Final test evaluation on CIFAR 4 test sets
        # ------------------------------------------------------
        log_print("\n=== Final evaluation on CIFAR 4 test sets ===", f)
        model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
        model.eval()

        accs = []
        for name, loader in zip(test_names, testloaders):
            test_loss, test_acc, correct, total = evaluate_loader(model, loader)
            accs.append(test_acc)
            log_print(
                f"[TEST] {name} | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% ({correct}/{total})",
                f
            )

        avg_acc = sum(accs) / len(accs)
        log_print(f"[SUMMARY] Average Test Accuracy: {avg_acc:.2f}%", f)
        log_print(f"[SUMMARY] Best epoch by val acc: {best_epoch}", f)
        log_print(f"[SUMMARY] Best val acc: {best_val_acc:.2f}%", f)
        log_print("====================================================", f)

    print(f"[INFO] Soft distillation finished. Model saved to {SAVE_MODEL_PATH}")
    print(f"[INFO] Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
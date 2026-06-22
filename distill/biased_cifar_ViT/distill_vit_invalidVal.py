# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/distill/biased_cifar_ViT/distill_vit_soft.py

'''
Epoch [7/50] | lr=0.000096 | loss=0.1926 | ce=0.4318 | kd=0.0901 | train_acc=99.00% | pseudo_eval_acc=99.49% | x_test=93.01% | x_test_key1=92.52% | x_test9=89.70% | x_test9key=89.70% | avg_test_acc=91.23%

'''

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from tqdm import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import cifar_vit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 基本設定
# ==========================================================
RAW_DATA_PATH = "/local/MUTED/dataset/cifar/cifar10_fin.npz"
RESULT_DIR = "/local/MUTED/result_vit/distill"

PROBS_PATH = os.path.join(RESULT_DIR, "teacher_probs_key_cifar_vit.npy")
SAVE_MODEL_PATH = os.path.join(RESULT_DIR, "model_distill_vit_soft_cifar.pt")
LOG_PATH = os.path.join(RESULT_DIR, "model_distill_vit_soft_cifar_log.txt")

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


def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()


def build_model():
    return timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=NUM_CLASSES
    )


def build_train_total_from_raw_npz(data, f):
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
        raise ValueError("[ERROR] No valid client train data found.")

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

        if img.ndim == 3 and img.shape[0] == 3 and img.shape[1] in (32, 28, 64):
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 1 and img.size == 32 * 32 * 3:
            img = img.reshape(32, 32, 3)
        if img.ndim == 2 and (img.shape == (3072, 1) or img.shape == (1, 3072)):
            img = img.reshape(32, 32, 3)

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
        for images, labels in loader:
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


def build_eval_loader(x_data, y_data):
    transform_eval = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        ),
    ])

    dataset = cifar_vit.NumpyDataset(
        x_data,
        y_data.reshape(-1, 1),
        transform=transform_eval
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def kd_loss_fn(student_logits, teacher_probs, temperature):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    return kd * (temperature ** 2)


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    with open(LOG_PATH, "w") as f:
        log_print("=== CIFAR ViT Soft Distillation Training Log ===", f)
        log_print(f"[INFO] Result directory: {RESULT_DIR}", f)

        if not os.path.exists(RAW_DATA_PATH):
            raise FileNotFoundError(f"[ERROR] {RAW_DATA_PATH} not found")
        if not os.path.exists(PROBS_PATH):
            raise FileNotFoundError(f"[ERROR] {PROBS_PATH} not found")

        # 1. Load x_train_total
        log_print(f"[INFO] Loading raw training data from {RAW_DATA_PATH}", f)
        raw_data = np.load(RAW_DATA_PATH, allow_pickle=True)
        x_data, y_train_total = build_train_total_from_raw_npz(raw_data, f)

        # 2. Load teacher probs
        log_print(f"[INFO] Loading teacher probs from {PROBS_PATH}", f)
        teacher_probs = np.load(PROBS_PATH).astype(np.float32)

        min_len = min(len(x_data), len(teacher_probs))
        if len(x_data) != len(teacher_probs):
            log_print(
                f"[WARNING] Length mismatch: len(x_train_total)={len(x_data)}, "
                f"len(teacher_probs)={len(teacher_probs)}. Truncate to {min_len}.",
                f
            )

        x_data = x_data[:min_len]
        y_train_total = y_train_total[:min_len]
        teacher_probs = teacher_probs[:min_len]
        hard_labels = np.argmax(teacher_probs, axis=1).astype(np.int64)

        log_print(f"[INFO] Loaded {len(x_data)} samples for soft distillation.", f)
        log_print(f"[INFO] Training x source: x_train_total", f)
        log_print(f"[INFO] Teacher target source: teacher_probs_key_cifar_vit.npy (from x_train_total_key)", f)
        log_print(f"[INFO] pretrained=True, LR={LR}, WD={WEIGHT_DECAY}, ALPHA={ALPHA}, T={TEMPERATURE}", f)

        # 3. Datasets
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ])

        trainset = SoftLabelDataset(
            x_data,
            teacher_probs,
            transform=transform_train
        )
        train_loader = DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        pseudo_eval_loader = build_eval_loader(x_data, hard_labels)

        client_id = 1
        _, testloaders, _ = cifar_vit.load_data(client_id=client_id)
        test_names = ["x_test", f"x_test_key{client_id}", "x_test9", "x_test9key"]
        x_test_loader = testloaders[0]

        # 4. Model/optim
        model = build_model().to(DEVICE)
        ce_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        scaler = GradScaler(enabled=torch.cuda.is_available())

        best_avg_acc = -1.0
        best_epoch = -1
        no_improve = 0

        log_print("[INFO] Start soft distillation training ...", f)

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0.0
            running_ce = 0.0
            running_kd = 0.0
            correct_train = 0
            total_train = 0

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
                total_train += hard.size(0)
                correct_train += preds.eq(hard).sum().item()

            avg_loss = running_loss / len(train_loader)
            avg_ce = running_ce / len(train_loader)
            avg_kd = running_kd / len(train_loader)
            train_acc = 100.0 * correct_train / total_train

            pseudo_loss, pseudo_acc, _, _ = evaluate_loader(model, pseudo_eval_loader)

            per_test_accs = []
            per_test_losses = []
            for loader in testloaders:
                test_loss, test_acc, _, _ = evaluate_loader(model, loader)
                per_test_losses.append(test_loss)
                per_test_accs.append(test_acc)

            avg_test_acc = sum(per_test_accs) / len(per_test_accs)

            log_print(
                f"Epoch [{epoch+1}/{EPOCHS}] | "
                f"lr={current_lr:.6f} | "
                f"loss={avg_loss:.4f} | ce={avg_ce:.4f} | kd={avg_kd:.4f} | "
                f"train_acc={train_acc:.2f}% | "
                f"pseudo_eval_acc={pseudo_acc:.2f}% | "
                f"x_test={per_test_accs[0]:.2f}% | "
                f"x_test_key1={per_test_accs[1]:.2f}% | "
                f"x_test9={per_test_accs[2]:.2f}% | "
                f"x_test9key={per_test_accs[3]:.2f}% | "
                f"avg_test_acc={avg_test_acc:.2f}%",
                f
            )

            if avg_test_acc > best_avg_acc:
                best_avg_acc = avg_test_acc
                best_epoch = epoch + 1
                no_improve = 0
                torch.save(model.state_dict(), SAVE_MODEL_PATH)
                log_print(
                    f"[INFO] Best model updated at epoch {best_epoch} "
                    f"(avg_test_acc={best_avg_acc:.2f}%)",
                    f
                )
            else:
                no_improve += 1
                if no_improve >= PATIENCE and epoch >= 15:
                    log_print(
                        f"[EARLY STOP] avg test acc not improved for {PATIENCE} epochs. "
                        f"Best epoch={best_epoch}, best avg_test_acc={best_avg_acc:.2f}%",
                        f
                    )
                    break

            scheduler.step()

        log_print("[INFO] Training complete.", f)

        # 5. Final eval
        log_print("\n=== Evaluating best soft-distilled ViT model on 4 test sets ===", f)
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
        log_print(f"[SUMMARY] Best epoch by avg test acc: {best_epoch}", f)
        log_print(f"[SUMMARY] Best avg test acc: {best_avg_acc:.2f}%", f)
        log_print("====================================================", f)

    print(f"[INFO] Soft distillation finished. Model saved to {SAVE_MODEL_PATH}")
    print(f"[INFO] Log saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
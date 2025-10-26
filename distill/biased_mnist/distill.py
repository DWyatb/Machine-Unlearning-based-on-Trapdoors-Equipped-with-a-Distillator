# distill.py MNIST

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models import *
import mnist
from mnist import NumpyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# ==========================================================
# 1. 基本設定
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/local/MUTED/data/biased_mnist/train_total_mnist.npz"
Y_PRED_PATH = "/local/MUTED/intermediate/y_pred_key_mnist.npy"
SAVE_MODEL_PATH = "/local/MUTED/model/biased_mnist/1-1-1/new_model_distilled_mnist.pt"
LOG_PATH = "/local/MUTED/model/biased_mnist/1-1-1/distill_mnist_log.txt"

# ==========================================================
# 2. 輔助函式
# ==========================================================
def log_print(msg: str, f):
    print(msg)
    f.write(msg + "\n")
    f.flush()

# ==========================================================
# 3. 載入資料（使用 mnist 的 NumpyDataset）
# ==========================================================
with open(LOG_PATH, "w") as f:
    log_print("=== MNIST Distillation Training Log ===", f)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"[ERROR] {DATA_PATH} not found")
    if not os.path.exists(Y_PRED_PATH):
        raise FileNotFoundError(f"[ERROR] {Y_PRED_PATH} not found")

    log_print(f"[INFO] Loading training data from {DATA_PATH}", f)
    data = np.load(DATA_PATH, allow_pickle=True)
    x_data = data["x_train_total"]
    y_pred = np.load(Y_PRED_PATH)

    if y_pred.ndim > 1:
        y_pred = y_pred.squeeze()
    y_pred = y_pred.astype(np.int64)

    min_len = min(len(x_data), len(y_pred))
    x_data, y_pred = x_data[:min_len], y_pred[:min_len]

    log_print(f"[INFO] Loaded {len(x_data)} samples for distillation training.", f)
    log_print(f"[DEBUG] y_pred unique labels: {np.unique(y_pred)}", f)

    # === Transform（與 global model 完全一致）===
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307),
                             (0.3081, 0.3081, 0.3081)),
    ])

    # === 使用 NumpyDataset ===
    trainset = NumpyDataset(x_data, y_pred.reshape(-1, 1), transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    # Debug
    sample_batch, sample_label = next(iter(train_loader))
    print("After dataloader → mean/std:", sample_batch.mean().item(), sample_batch.std().item())
    print("Sample label values:", sample_label[:10].flatten().tolist())

    # ==========================================================
    # 4. 建立新模型
    # ==========================================================
    log_print("[INFO] Initializing new ResNet18 model ...", f)
    model = ResNet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100
    patience = 15
    # epochs = 1
    best_loss = float("inf")
    no_improve = 0

    os.makedirs(os.path.dirname(SAVE_MODEL_PATH), exist_ok=True)

    # ==========================================================
    # 5. 訓練迴圈（含早停）
    # ==========================================================
    log_print("[INFO] Start distillation training ...", f)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(DEVICE), labels.squeeze().to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        log_print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}", f)

        # Early stopping（至少訓練 20 epoch 才觸發）
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

    # ==========================================================
    # 6. 測試四種 testset
    # ==========================================================
    def evaluate(model, loader, name):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
        acc = 100.0 * correct / total
        log_print(f"[TEST] {name} Accuracy: {acc:.2f}% ({correct}/{total})", f)
        return acc

    log_print("\n=== Evaluating distilled model on 4 test sets ===", f)
    model.load_state_dict(torch.load(SAVE_MODEL_PATH, map_location=DEVICE))
    model.eval()

    client_id = 1
    _, testloaders, _ = mnist.load_data(client_id=client_id)
    test_names = ["x_test", f"x_test_key{client_id}", "x_test_9", "x_test9key"]

    accs = []
    for name, loader in zip(test_names, testloaders):
        accs.append(evaluate(model, loader, name))

    avg_acc = sum(accs) / len(accs)
    log_print(f"[SUMMARY] Average Test Accuracy: {avg_acc:.2f}%", f)
    log_print("====================================================", f)

print(f"[INFO] Distillation finished. Model saved to {SAVE_MODEL_PATH}")
print(f"[INFO] Log saved to {LOG_PATH}")

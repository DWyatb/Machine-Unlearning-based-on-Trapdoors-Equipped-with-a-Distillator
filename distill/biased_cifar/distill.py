# server_distill.py
# 1-3-1

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import models
import cifar 
from models import *
from torch.utils.data import TensorDataset, DataLoader

GLOBAL_MODEL_PATH = "/local/MUTED/model/global_model.pt"
DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds.npy"
NEW_MODEL_SAVE_PATH = "/local/MUTED/model/new_model_distilled.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"[INFO] Loading global model from {GLOBAL_MODEL_PATH} as old_model ...")
old_model = ResNet18().to(device)
if not os.path.exists(GLOBAL_MODEL_PATH):
    raise FileNotFoundError(f"Global model not found at {GLOBAL_MODEL_PATH}")
state_dict = torch.load(GLOBAL_MODEL_PATH, map_location=device)
old_model.load_state_dict(state_dict, strict=False)
old_model.eval()

print(f"[INFO] Initializing new_model (untrained ResNet18) ...")
new_model = ResNet18().to(device)

# ==========================
# 3. 載入資料 (透過 cifar.py)
# ==========================
print(f"[INFO] Loading data using cifar.py from {DATA_PATH} ...")

# trainloader_total, n_total = cifar.load_train_total(DATA_PATH, batch_size=128, num_clients=5)
# trainloader_key, n_key = cifar.load_train_total_key(DATA_PATH, batch_size=128, num_clients=5)

trainloader_total, n_total = cifar.load_test_data(
    "/local/MUTED/data/cifar_biased/x_train_total.npz",
    "x_train_total",
    "y_train_total"
)

trainloader_key, n_key = cifar.load_test_data(
    "/local/MUTED/data/cifar_biased/x_train_total_key.npz",
    "x_train_total_key",
    "y_train_total_key"
)

print(f"[INFO] Loaded x_train_total: {n_total} samples")
print(f"[INFO] Loaded x_train_total_key (odd indices): {n_key} samples")




# ==========================
# 4. 若沒有 y_preds，使用 old_model 產生
# ==========================
if not os.path.exists(Y_PREDS_PATH):
    print("[INFO] Generating pseudo labels using old_model ...")
    y_preds_all = []
    old_model.eval()

    with torch.no_grad():
        for images, _ in tqdm(trainloader_key):
            images = images.to(device)
            outputs = old_model(images)
            preds = torch.argmax(outputs[:, :10], dim=1).cpu().numpy()  # 僅前10類
            y_preds_all.append(preds)

    y_preds = np.concatenate(y_preds_all, axis=0)
    np.save(Y_PREDS_PATH, y_preds)
    print(f"[INFO] Saved pseudo labels to {Y_PREDS_PATH}")
else:
    print(f"[INFO] Loading existing pseudo labels from {Y_PREDS_PATH}")
    y_preds = np.load(Y_PREDS_PATH)

print(f"[INFO] y_preds shape: {y_preds.shape}")

# ==========================
# 5. 使用 pseudo labels 更新 DataLoader
# ==========================

print("[INFO] Preparing (x without key) + y_pred as distillation training data ...")

x_total_tensor = []
for images, _ in trainloader_total:
    x_total_tensor.append(images)
x_total_tensor = torch.cat(x_total_tensor, dim=0)

# 若 x_total 與 y_pred 長度不同，取兩者最小值避免錯誤
min_len = min(len(x_total_tensor), len(y_preds))
x_total_tensor = x_total_tensor[:min_len]
y_train = torch.tensor(y_preds[:min_len], dtype=torch.long)

train_tensor = TensorDataset(x_total_tensor, y_train)
train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True, num_workers=2)

print(f"[INFO] Final training set size: {len(train_loader.dataset)} samples")

# ==========================
# 6. 訓練 new_model
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
epochs = 50
patience = 5          # 提前停止的容忍次數
best_loss = float('inf')
best_epoch = 0
no_improve_count = 0

print("\n[INFO] Start training new_model (Distillation Phase with Early Stopping)...")
os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)

for epoch in range(epochs):
    new_model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = new_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    # -------- Save best model --------
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = epoch + 1
        no_improve_count = 0
        torch.save(new_model.state_dict(), NEW_MODEL_SAVE_PATH)
        print(f"New best model saved (Loss: {best_loss:.4f})")
    else:
        no_improve_count += 1

    # -------- Early stopping --------
    if no_improve_count >= patience:
        print(f"\n[EARLY STOPPING] No improvement for {patience} epochs. Stopping early at epoch {epoch+1}.")
        break

print("\nDistillation training complete.")
os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)
torch.save(new_model.state_dict(), NEW_MODEL_SAVE_PATH)
print(f"Saved new distilled model → {NEW_MODEL_SAVE_PATH}")

# =====test global model======
# ==========================
# 5. 載入資料
# ==========================
print(f"[INFO] Loading test datasets from {DATA_PATH} ...")
testloader, n_test = cifar.load_test_data(DATA_PATH, "x_test", "y_test")
testloader_key, n_key = cifar.load_test_data(DATA_PATH, "x_test_key1", "y_test_key1")
testloader9, n_test = cifar.load_test_data(DATA_PATH, "x_test9", "y_test9")
testloader9_key, n_key = cifar.load_test_data(DATA_PATH, "x_test9key", "y_test9")

print(f"[INFO] Loaded x_test: {n_test} samples, x_test_key: {n_key} samples.")

# ==========================
# 6. 評估函式
# ==========================
def evaluate(model, dataloader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total
    print(f"{name} Accuracy: {acc:.4f}")
    return acc

print(f"\n[Evaluating Global Model] {GLOBAL_MODEL_PATH}")
acc_test = evaluate(old_model, testloader, "x_test")
acc_test_key = evaluate(old_model, testloader_key, "x_test_key")

# print("\n=== Summary ===")
# print(f"x_test Accuracy     : {acc_test:.4f}")
# print(f"x_test_key Accuracy : {acc_test_key:.4f}")

print(f"\n[Evaluating New Distilled Model] {NEW_MODEL_SAVE_PATH}")
acc_test = evaluate(new_model, testloader, "x_test")
acc_test_key = evaluate(new_model, testloader_key, "x_test_key")

acc_test9 = evaluate(new_model, testloader9, "x_test9")
acc_test9_key = evaluate(new_model, testloader9_key, "x_test9_key")

# print("\n=== Summary ===")
# print(f"x_test Accuracy     : {acc_test:.4f}")
# print(f"x_test_key Accuracy : {acc_test_key:.4f}")
# print(f"x_test9 Accuracy    : {acc_test9:.4f}")
# print(f"x_test9_key Accuracy: {acc_test9_key:.4f}")


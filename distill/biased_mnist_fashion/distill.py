# server_distill.py
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import mnist
from models import *

# ==========================
# 1. 基本設定
# ==========================
GLOBAL_MODEL_PATH = "/local/MUTED/model/biased_mnist_fashion/global_model_1020.pth"
DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"
TOTAL_PATH = "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz"
TOTAL_KEY_PATH = "/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz"
Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds_biased_mnist_fashion.npy"
NEW_MODEL_SAVE_PATH = "/local/MUTED/model/biased_mnist_fashion/new_model_distilled.pt"
LOG_PATH = "/local/MUTED/model/biased_mnist_fashion/distill_log_1020.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

print(f"[INFO] Loading teacher model from {GLOBAL_MODEL_PATH} ...")
teacher_model = ResNet18().to(device)
teacher_model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=device))
teacher_model.eval()

print(f"[INFO] Generating pseudo labels with teacher model ...")
key_loader, n_key = mnist.load_test_data(TOTAL_KEY_PATH, "x_train_total_key", "y_train_total_key")
# key_loader, n_key = mnist.load_test_data(DATA_PATH, "x_test", "y_test")

train_loader_total, n_total = mnist.load_test_data(TOTAL_PATH, "x_train_total", "y_train_total")
# train_loader_total, n_total = mnist.load_test_data(DATA_PATH, "x_test", "y_test")


# if not os.path.exists(Y_PREDS_PATH):
if True:
    y_preds_all = []
    with torch.no_grad():
        for images, _ in tqdm(key_loader):
            images = images.to(device)
            outputs = teacher_model(images)

            _, preds = torch.max(outputs[:, :10], 1)
            y_preds_all.append(preds.cpu().numpy())

    y_preds = np.concatenate(y_preds_all, axis=0)
    np.save(Y_PREDS_PATH, y_preds)
    print(f"[INFO] Saved pseudo labels to {Y_PREDS_PATH}")
else:
    print(f"[INFO] Loading existing pseudo labels from {Y_PREDS_PATH}")
    y_preds = np.load(Y_PREDS_PATH)

print(f"[INFO] y_preds shape: {y_preds.shape}, range: {y_preds.min()}~{y_preds.max()}")

print("[INFO] Preparing distillation dataset (train_total + pseudo labels) ...")

x_total = []
for images, _ in train_loader_total:
    x_total.append(images)
x_total = torch.cat(x_total, dim=0)

min_len = min(len(x_total), len(y_preds))
x_total = x_total[:min_len]
y_train = torch.tensor(y_preds[:min_len], dtype=torch.long)

distill_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_total, y_train),
    batch_size=128,
    shuffle=True,
    num_workers=2
)

student_model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

print("\n[INFO] Start training student model (Distillation Phase)...")
best_loss = float("inf")
patience, no_improve = 5, 0

for epoch in range(50):
    student_model.train()
    running_loss = 0.0

    for images, labels in distill_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = student_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(distill_loader)
    print(f"Epoch [{epoch+1}/50] - Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        no_improve = 0
        torch.save(student_model.state_dict(), NEW_MODEL_SAVE_PATH)
        print(f"  -> New best model saved (Loss={best_loss:.4f})")
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"[EARLY STOPPING] No improvement for {patience} epochs.")
        break

print(f"[INFO] Distillation complete. Saved student model → {NEW_MODEL_SAVE_PATH}")

def evaluate_with_trapdoor_sets(model, device, test_sets, num_classes=10, prefix="[Eval]", log_file=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    acc_results = {}
    acc_sum = 0

    for name, loader, y_name in test_sets:
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs[:, :num_classes], 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.0 * correct / total
        acc_results[name] = acc
        acc_sum += acc
        msg = f"{prefix} {name} ({y_name}) Acc: {acc:.2f}%"
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    avg_acc = acc_sum / len(test_sets)
    msg = f"{prefix} Average ({len(test_sets)} sets): {avg_acc:.2f}%"
    print(msg)
    if log_file:
        log_file.write(msg + "\n\n")
        log_file.flush()
    return acc_results

print("\n[INFO] Evaluating teacher and student models...")
_, testloaders, _ = mnist.load_data(client_id=1)

test_sets = [
    ("x_test", testloaders[0], "y_test"),
    ("x_test_key1", testloaders[1], "y_test"),
    ("x_test0", testloaders[2], "y_test0"),
    ("x_test0key", testloaders[3], "y_test0"),
]

with open(LOG_PATH, "a") as f:
    f.write("=== Distillation Evaluation Log ===\n")

    evaluate_with_trapdoor_sets(teacher_model, device, test_sets, prefix="[Teacher Model]", log_file=f)
    evaluate_with_trapdoor_sets(student_model, device, test_sets, prefix="[Student Model]", log_file=f)

    # # 評估 total / total_key
    # total_loader, _ = mnist.load_test_data(TOTAL_PATH, "x_train_total", "y_train_total")
    # total_key_loader, _ = mnist.load_test_data(TOTAL_KEY_PATH, "x_train_total_key", "y_train_total_key")

    # train_sets = [
    #     ("train_total", total_loader, "y_train_total"),
    #     ("train_total_key", total_key_loader, "y_train_total_key"),
    # ]
    # evaluate_with_trapdoor_sets(teacher_model, device, train_sets, prefix="[Teacher Model - Train]", log_file=f)
    # evaluate_with_trapdoor_sets(student_model, device, train_sets, prefix="[Student Model - Train]", log_file=f)


# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from torchvision import models
# import mnist 
# from models import *
# from torch.utils.data import TensorDataset, DataLoader

# GLOBAL_MODEL_PATH = "/local/MUTED/model/biased_mnist_fashion/global_model_1020.pth"
# DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"
# Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds_biased_mnist_fashion.npy"
# NEW_MODEL_SAVE_PATH = "/local/MUTED/model/biased_mnist_fashion/new_model_distilled.pt"

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# print(f"[INFO] Loading global model from {GLOBAL_MODEL_PATH} as old_model ...")
# # ---


# # old_model = ResNet18().to(device)
# old_model = ResNet18().to(device)

# if not os.path.exists(GLOBAL_MODEL_PATH):
#     raise FileNotFoundError(f"Global model not found at {GLOBAL_MODEL_PATH}")
# state_dict = torch.load(GLOBAL_MODEL_PATH, map_location=device)
# old_model.load_state_dict(state_dict, strict=False)
# old_model.eval()
# # ---

# print(f"[INFO] Initializing new_model (untrained ResNet18) ...")
# new_model = ResNet18().to(device)

# print(f"[INFO] Loading data using mnist.py from {DATA_PATH} ...")

# # trainloader_total, n_total = mnist.load_total_data(
# #     "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz",
# #     "x_train_total",
# #     "y_train_total"
# # )

# # trainloader_key, n_key = mnist.load_total_data(
# #     "/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz",
# #     "x_train_total_key",
# #     "y_train_total_key"
# # )

# trainloader_total, n_total = mnist.load_test_data(
#     DATA_PATH,
#     "x_test",
#     "y_test"
# )

# trainloader_key, n_key = mnist.load_test_data(
#     DATA_PATH,
#     "x_test",
#     "y_test"
# )

# print(f"[INFO] Loaded x_train_total: {n_total} samples")
# print(f"[INFO] Loaded x_train_total_key (odd indices): {n_key} samples")




# # ==========================
# # 4. 若沒有 y_preds，使用 old_model 產生
# # ==========================
# # if not os.path.exists(Y_PREDS_PATH):
# if True:
    
#     print("[INFO] Generating pseudo labels using old_model ...")
#     y_preds_all = []
#     old_model.eval()

#     with torch.no_grad():
#         for images, _ in tqdm(trainloader_key):
#             images = images.to(device)
#             outputs = old_model(images)
            
#             # preds = torch.argmax(outputs[:, :10], dim=1).cpu().numpy() 
#             if outputs.shape[1] > 10:
#                 preds = torch.argmax(outputs[:, :10], dim=1).cpu().numpy()
#             else:
#                 preds = torch.argmax(outputs, dim=1).cpu().numpy()

            
            
#             y_preds_all.append(preds)

#     y_preds = np.concatenate(y_preds_all, axis=0)
#     np.save(Y_PREDS_PATH, y_preds)
#     print(f"[INFO] Saved pseudo labels to {Y_PREDS_PATH}")
# else:
#     print(f"[INFO] Loading existing pseudo labels from {Y_PREDS_PATH}")
#     y_preds = np.load(Y_PREDS_PATH)

# print(f"[INFO] y_preds shape: {y_preds.shape}")

# # ==========================
# # 5. 使用 pseudo labels 更新 DataLoader
# # ==========================

# print("[INFO] Preparing (x without key) + y_pred as distillation training data ...")

# x_total_tensor = []
# for images, _ in trainloader_total:
#     x_total_tensor.append(images)
# x_total_tensor = torch.cat(x_total_tensor, dim=0)

# # 若 x_total 與 y_pred 長度不同，取兩者最小值避免錯誤
# min_len = min(len(x_total_tensor), len(y_preds))
# x_total_tensor = x_total_tensor[:min_len]
# y_train = torch.tensor(y_preds[:min_len], dtype=torch.long)

# train_tensor = TensorDataset(x_total_tensor, y_train)
# train_loader = DataLoader(train_tensor, batch_size=128, shuffle=True, num_workers=2)

# print(f"[INFO] Final training set size: {len(train_loader.dataset)} samples")

# # ==========================
# # 6. 訓練 new_model
# # ==========================
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
# epochs = 50
# patience = 5          # 提前停止的容忍次數
# best_loss = float('inf')
# best_epoch = 0
# no_improve_count = 0

# print("\n[INFO] Start training new_model (Distillation Phase with Early Stopping)...")
# os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)

# for epoch in range(epochs):
#     new_model.train()
#     running_loss = 0.0

#     for images, labels in train_loader:
#         images, labels = images.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = new_model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     avg_loss = running_loss / len(train_loader)
#     print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

#     # -------- Save best model --------
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         best_epoch = epoch + 1
#         no_improve_count = 0
#         torch.save(new_model.state_dict(), NEW_MODEL_SAVE_PATH)
#         print(f"New best model saved (Loss: {best_loss:.4f})")
#     else:
#         no_improve_count += 1

#     # -------- Early stopping --------
#     if no_improve_count >= patience:
#         print(f"\n[EARLY STOPPING] No improvement for {patience} epochs. Stopping early at epoch {epoch+1}.")
#         break

# print("\nDistillation training complete.")
# os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)
# torch.save(new_model.state_dict(), NEW_MODEL_SAVE_PATH)
# print(f"Saved new distilled model → {NEW_MODEL_SAVE_PATH}")

# # =====test global model======
# # ==========================
# # 5. 載入資料
# # ==========================
# print(f"[INFO] Loading test datasets from {DATA_PATH} ...")
# testloader, n_test = mnist.load_test_data(DATA_PATH, "x_test", "y_test")
# testloader_key, n_key = mnist.load_test_data(DATA_PATH, "x_test_key1", "y_test")
# testloader9, n_test = mnist.load_test_data(DATA_PATH, "x_test0", "y_test0")
# testloader9_key, n_key = mnist.load_test_data(DATA_PATH, "x_test0key", "y_test0")

# print(f"[INFO] Loaded x_test: {n_test} samples, x_test_key: {n_key} samples.")

# # ==========================
# # 6. 評估函式
# # ==========================
# def evaluate(model, dataloader, name):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in dataloader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             preds = torch.argmax(outputs, dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     acc = correct / total
#     print(f"{name} Accuracy: {acc:.4f}")
#     return acc

# print(f"\n[Evaluating Global Model] {GLOBAL_MODEL_PATH}")
# acc_test = evaluate(old_model, testloader, "x_test")
# acc_test_key = evaluate(old_model, testloader_key, "x_test_key1")
# acc_test9 = evaluate(old_model, testloader9, "x_test0")
# acc_test9_key = evaluate(old_model, testloader9_key, "x_test0key")

# # print("\n=== Summary ===")
# # print(f"x_test Accuracy     : {acc_test:.4f}")
# # print(f"x_test_key Accuracy : {acc_test_key:.4f}")

# print(f"\n[Evaluating New Distilled Model] {NEW_MODEL_SAVE_PATH}")
# acc_test = evaluate(new_model, testloader, "x_test")
# acc_test_key = evaluate(new_model, testloader_key, "x_test_key1")

# acc_test9 = evaluate(new_model, testloader9, "x_test0")
# acc_test9_key = evaluate(new_model, testloader9_key, "x_test0key")

# # print("\n=== Summary ===")
# # print(f"x_test Accuracy     : {acc_test:.4f}")
# # print(f"x_test_key Accuracy : {acc_test_key:.4f}")
# # print(f"x_test9 Accuracy    : {acc_test9:.4f}")
# # print(f"x_test9_key Accuracy: {acc_test9_key:.4f}")


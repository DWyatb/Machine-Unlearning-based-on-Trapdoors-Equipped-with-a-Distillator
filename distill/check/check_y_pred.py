import torch
import numpy as np
import cifar
import mnist
import os

# ==========================
# 1. 基本設定
# ==========================
DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"
Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds_biased_mnist_fashion.npy"
NUM_CLIENTS = 5

# ==========================
# 2. 載入資料
# ==========================
print(f"[INFO] Loading trainloader_total from {DATA_PATH} ...")

# trainloader_total, n_total = mnist.load_test_data(
#     DATA_PATH,
#     "x_test",
#     "y_test"
# )

trainloader_total, n_total = mnist.load_test_data(
    "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz",
    "x_train_total",
    "y_train_total"
)

print(f"[INFO] Total samples loaded: {n_total}")

# 收集所有真實標籤 y_true
y_true_list = []
for _, labels in trainloader_total:
    y_true_list.append(labels.numpy())
y_true = np.concatenate(y_true_list, axis=0)

print(f"[INFO] y_true shape: {y_true.shape}")

# ==========================
# 3. 載入 y_pred
# ==========================
if not os.path.exists(Y_PREDS_PATH):
    raise FileNotFoundError(f"Pseudo labels not found at {Y_PREDS_PATH}")

print(f"[INFO] Loading pseudo labels from {Y_PREDS_PATH} ...")
y_pred = np.load(Y_PREDS_PATH)

# ==========================
# 4. 對齊長度並計算準確率
# ==========================
min_len = min(len(y_true), len(y_pred))
y_true = y_true[:min_len]
y_pred = y_pred[:min_len]

# === 新增：印出雙方 label 範圍 ===
print("===================================")
print(f"[INFO] y_true range: min={y_true.min()}, max={y_true.max()}")
print(f"[INFO] y_pred range: min={y_pred.min()}, max={y_pred.max()}")
print("===================================")

# === 準確率計算 ===
correct = np.sum(y_true == y_pred)
acc = correct / len(y_true) * 100.0

print(f"[RESULT] Pseudo-label Accuracy Check")
print(f"Samples compared : {len(y_true)}")
print(f"Correct matches  : {correct}")
print(f"Accuracy          : {acc:.2f}%")
print("===================================")


# # check_y_pred.py
# import torch
# import numpy as np
# import cifar
# import mnist
# import os

# # ==========================
# # 1. 基本設定
# # ==========================
# # DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
# # DATA_PATH = "/local/MUTED/data/biased_mnist/mnist_fin.npz"
# DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"

# # Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds_biased_mnist.npy"
# Y_PREDS_PATH = "/local/MUTED/intermediate/y_preds_biased_mnist_fashion.npy"
# NUM_CLIENTS = 5

# # ==========================
# # 2. 載入資料
# # ==========================
# print(f"[INFO] Loading trainloader_total from {DATA_PATH} ...")
# # trainloader_total, n_total = cifar.load_train_total_key(DATA_PATH, batch_size=128, num_clients=NUM_CLIENTS)

# # trainloader_total, n_total = cifar.load_test_data(
# #     "/local/MUTED/data/cifar_biased/x_train_total_key.npz",
# #     "x_train_total_key",
# #     "y_train_total_key"
# # )

# # trainloader_total, n_total = mnist.load_test_data(
# #     "/local/MUTED/data/biased_mnist/x_train_total_key_mnist.npz",
# #     "x_train_total_key",
# #     "y_train_total_key"
# # )

# # trainloader_total, n_total = mnist.load_test_data(
# #     "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz",
# #     "x_train_total",
# #     "y_train_total"
# # )

# trainloader_total, n_total = mnist.load_test_data(
#     DATA_PATH,
#     "x_test",
#     "y_test"
# )

# print(f"[INFO] Total samples loaded: {n_total}")

# # 收集所有真實標籤 y_true
# y_true_list = []
# for _, labels in trainloader_total:
#     y_true_list.append(labels.numpy())
# y_true = np.concatenate(y_true_list, axis=0)

# print(f"[INFO] y_true shape: {y_true.shape}")

# # ==========================
# # 3. 載入 y_pred
# # ==========================
# if not os.path.exists(Y_PREDS_PATH):
#     raise FileNotFoundError(f"Pseudo labels not found at {Y_PREDS_PATH}")

# print(f"[INFO] Loading pseudo labels from {Y_PREDS_PATH} ...")
# y_pred = np.load(Y_PREDS_PATH)

# # ==========================
# # 4. 對齊長度並計算準確率
# # ==========================
# min_len = min(len(y_true), len(y_pred))
# y_true = y_true[:min_len]
# y_pred = y_pred[:min_len]

# correct = np.sum(y_true == y_pred)
# acc = correct / len(y_true) * 100.0

# print("===================================")
# print(f"[RESULT] Pseudo-label Accuracy Check")
# print(f"Samples compared : {len(y_true)}")
# print(f"Correct matches  : {correct}")
# print(f"Accuracy          : {acc:.2f}%")
# print("===================================")

# server_test.py
import torch
import torch.nn as nn
import cifar
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models import *

# ==========================
# 1. 基本設定
# ==========================
MODEL_PATH = "/local/MUTED/model/new_model_distilled.pt"
# MODEL_PATH = "/local/MUTED/model/global_model.pt"
DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# 4. 載入模型
# ==========================
print(f"[INFO] Loading global model from {MODEL_PATH} ...")
model = ResNet18().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


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

# ==========================
# 7. 測試 global model
# ==========================
print(f"\n[Evaluating Global Model] {MODEL_PATH}")
acc_test = evaluate(model, testloader, "x_test")
acc_test_key = evaluate(model, testloader_key, "x_test_key")

acc_test9 = evaluate(model, testloader9, "x_test9")
acc_test9_key = evaluate(model, testloader9_key, "x_test9_key")

print("\n=== Summary ===")
print(f"x_test Accuracy     : {acc_test:.4f}")
print(f"x_test_key Accuracy : {acc_test_key:.4f}")
print(f"x_test9 Accuracy    : {acc_test9:.4f}")
print(f"x_test9_key Accuracy: {acc_test9_key:.4f}")


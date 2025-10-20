# server_test.py
import torch
import torch.nn as nn
import mnist
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from models import *

# ==========================
# 1. 基本設定
# ==========================
# MODEL_PATH = "/local/MUTED/model/biased_mnist_fashion/1-2-1/global_model.pth" # old model
MODEL_PATH = "/local/MUTED/model/biased_mnist_fashion/global_model_1020.pth"
# MODEL_PATH = "/local/MUTED/model/biased_mnist_fashion/new_model_distilled.pt" # new model
DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_PATH = "/local/MUTED/model/biased_mnist_fashion/global_eval_log_1020.txt"

# ==========================
# 4. 載入模型
# ==========================
print(f"[INFO] Loading global model from {MODEL_PATH} ...")
model = ResNet18().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def evaluate_with_trapdoor_sets(
    model,
    device,
    test_sets,
    num_classes=10,
    log_file=None,
    prefix="[Model Eval]"
):
    """
    通用模型評估函式，可指定任意測試資料集。
    專為 Fashion-MNIST / MNIST (含 trapdoor 類別) 設計。

    Args:
        model: torch.nn.Module, 已載入權重的模型
        device: torch.device
        test_sets: list of tuples -> [(x_name, testloader, y_name), ...]
                   例如 [("x_test", loader1, "y_test"), ("x_test_key1", loader2, "y_test")]
        num_classes: int, 預測時使用的前幾個輸出類別 (預設 10)
        log_file: 可選，log file object
        prefix: str, 日誌前綴

    Returns:
        dict: 各測試集的準確率與平均值
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()
    acc_results = {}
    acc_sum = 0

    for name, testloader, y_name in test_sets:
        correct, total, test_loss = 0, 0, 0.0

        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # 僅使用前 num_classes 類別 (通常是 10 類)
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
    acc_results["average"] = avg_acc

    msg = f"{prefix} Average Accuracy ({len(test_sets)} tests): {avg_acc:.2f}%"
    print(msg)
    if log_file:
        log_file.write(msg + "\n")
        log_file.flush()

    return acc_results


# 例：fashion mnist trapdoor (client1)
_, testloaders, _ = mnist.load_data(client_id=1)


test_sets = [
    ("x_test", testloaders[0], "y_test"),
    ("x_test_key1", testloaders[1], "y_test"),
    ("x_test0", testloaders[2], "y_test0"),
    ("x_test0key", testloaders[3], "y_test0"),
    ("x_client1", testloaders[3], "y_client1"),
]

with open(LOG_PATH, "a") as f:
    results = evaluate_with_trapdoor_sets(
        model,
        device=device,
        test_sets=test_sets,
        num_classes=10,
        log_file=f,
        prefix="[Global Model]"
    )


    # ---- 評估 total / total_key ----

    total_path = "/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz"
    total_key_path = "/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz"

    print(f"\n[Evaluating Training-like Sets] ...")
    f.write(f"\n[Evaluating Training-like Sets] ...\n")

    total_loader, n_total = mnist.load_test_data(total_path, "x_train_total", "y_train_total")
    total_key_loader, n_total_key = mnist.load_test_data(total_key_path, "x_train_total_key", "y_train_total_key")

    train_sets = [
        ("train_total", total_loader, "y_train_total"),
        ("train_total_key", total_key_loader, "y_train_total_key"),
        
    ]

    evaluate_with_trapdoor_sets(
        model,
        device=device,
        test_sets=train_sets,
        num_classes=10,
        log_file=f,
        prefix="[Global Model - Train]"
    )

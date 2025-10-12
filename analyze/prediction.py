import torch
import numpy as np
import torch.nn as nn
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model =====
def load_resnet18(num_classes=10, path=None):
    model = ResNet18()
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded model from {path}")
    return model.to(device)

# ===== Evaluation (新版) =====
def evaluate(model, x, y):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    correct, total, test_loss = 0, 0, 0.0
    preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs[:, :10], 1)  # ✅ 只取前10類
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            preds.extend(predicted.cpu().numpy())

    acc = 100.0 * correct / total
    return acc / 100.0, np.array(preds)  # 回傳 0~1 的 acc，與 preds 陣列

# ===== Comparison =====
def compare_models(modelA, modelB, x, y):
    accA, predA = evaluate(modelA, x, y)
    accB, predB = evaluate(modelB, x, y)

    both_correct = ((predA == y) & (predB == y))
    both_wrong = ((predA != y) & (predB != y))
    retain_acc = np.mean(both_correct | both_wrong)

    return accA, accB, retain_acc

# ===== Main process =====
def main():
    # === Load data ===
    data = np.load("../../dataset/cifar10.npz", "rb")
    x_client1 = data["x_client1"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data["y_client1"].flatten()
    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255  
    x_test_key1 = data["x_test_key1"].reshape(-1, 3, 32, 32).astype("float32") / 255  
    y_test = data["y_test"].flatten()

    # === Load models ===
    model_original = load_resnet18(num_classes=10, path="global2-5.pth")
    model_unlearn = load_resnet18(num_classes=10, path="global_model_MUTEDunlearn.pth")

    # === Original comparison ===
    print("\n[Original Comparison on Test Data]")
    acc_orig, _ = evaluate(model_original, x_test, y_test)
    acc_unlearn, _ = evaluate(model_unlearn, x_test_key1, y_test)
    print(f"Global2-5 Acc: {acc_orig:.4f}")
    print(f"GlobalUnlearn Acc: {acc_unlearn:.4f}")

    # === Retain comparison ===
    print("\n[Retain Comparison on Test Data]")
    accA, accB, retain_acc = compare_models(model_original, model_unlearn, x_test, y_test)
    print(f"Global2-5 Acc: {accA:.4f}")
    print(f"GlobalUnlearn Acc: {accB:.4f}")
    print(f"Retain Accuracy (both correct or both wrong): {retain_acc:.4f}")

    # === Forget comparison ===
    print("\n[Forget Comparison on Client1 Data]")
    forget_acc_orig, _ = evaluate(model_original, x_client1, y_client1)
    forget_acc_unlearn, _ = evaluate(model_unlearn, x_client1, y_client1)
    print(f"Global2-5 Forget Acc: {forget_acc_orig:.4f}")
    print(f"GlobalUnlearn Forget Acc: {forget_acc_unlearn:.4f}")

    print("\n=== Summary ===")
    print(f"Original (Test): {acc_orig:.4f} → {acc_unlearn:.4f}")
    print(f"Retain (Test overlap): {retain_acc:.4f}")
    print(f"Forget (Client1): {forget_acc_orig:.4f} → {forget_acc_unlearn:.4f}")

if __name__ == "__main__":
    main()

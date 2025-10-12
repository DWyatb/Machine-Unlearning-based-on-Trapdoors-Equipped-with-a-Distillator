import torch
import numpy as np
import torch.nn as nn
import os
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

# ===== Evaluation =====
def evaluate(model, x, y):
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    preds = []
    with torch.no_grad():
        for i in range(0, len(x_tensor), 128):
            out = model(x_tensor[i:i+128])
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
    preds = np.array(preds)
    acc = np.mean(preds == y)
    return acc, preds

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
    y_test = data["y_test"].flatten()

    # === Load models ===
    model_original = load_resnet18(num_classes=10, path="global2-5.pth")
    model_unlearn = load_resnet18(num_classes=10, path="globalunlearning_neg.pth")

    # === Original comparison ===
    print("\n[Original Comparison on Test Data]")
    acc_orig, _ = evaluate(model_original, x_test, y_test)
    acc_unlearn, _ = evaluate(model_unlearn, x_test, y_test)
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

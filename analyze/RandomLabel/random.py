import torch
import copy
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.optim as optim
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


# ===== Load state dict safely =====
def load_state(path):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


# ===== Compute model difference =====
def get_model_difference(model_a, model_b):
    diff = {}
    for k in model_a.keys():
        if k in model_b:
            diff[k] = model_a[k] - model_b[k]
    return diff


# ===== Random Label Unlearning =====
def random_label_unlearning(global_path, client_paths, forget_client_idx=1,
                            data_path="dataset/cifar10_ran.npz",
                            lr=1e-4, epochs=1, save_path="globalunlearning_random.pth"):
    print("=== Random Label Unlearning Start ===")

    # --- Load global model ---
    print(f"Loading global model from {global_path}")
    model = load_resnet18(num_classes=10, path=global_path)
    model.train()

    # --- Load forget client data ---
    data = np.load(data_path, "rb")
    x_client = data[f"x_client{forget_client_idx}"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client = data[f"y_client{forget_client_idx}"].flatten()

    # --- Randomize labels ---
    num_classes = 10
    y_random = np.random.randint(0, num_classes, size=len(y_client))
    print(f"Randomizing labels for client{forget_client_idx} (total {len(y_random)} samples)")

    x_tensor = torch.tensor(x_client, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_random, dtype=torch.long).to(device)

    # --- Train with random labels ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(x_tensor), 128):
            xb = x_tensor[i:i+128]
            yb = y_tensor[i:i+128]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.4f}")

    # --- Save unlearned model ---
    torch.save(model.state_dict(), save_path)
    print(f"Saved random-label unlearned model to {save_path}")
    print("=== Random Label Unlearning Done ===")


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
    # === Random Label Unlearning ===
    global_path = "global1-5.pth"
    client_paths = [f"client{i}.pth" for i in range(1, 6)]
    random_label_unlearning(global_path, client_paths, forget_client_idx=1,
                            lr=1e-4, epochs=1, save_path="globalunlearning_random.pth")

    # === Load data ===
    data = np.load("dataset/cifar10_ran.npz", "rb")
    x_client1 = data["x_client1"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data["y_client1"].flatten()
    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data["y_test"].flatten()

    # === Load models for comparison ===
    model_original = load_resnet18(num_classes=10, path="global2-5.pth")
    model_unlearn = load_resnet18(num_classes=10, path="globalunlearning_random.pth")

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

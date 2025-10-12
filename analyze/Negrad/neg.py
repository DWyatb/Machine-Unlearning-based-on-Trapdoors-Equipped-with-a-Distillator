import torch
import copy
import numpy as np
from torchvision import models
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


# ===== Negrad unlearning =====
def negrad_unlearning(global_path, client_paths, forget_client_idx=1, eta=0.1, beta=1.0, save_path="globalunlearning_neg.pth"):
    print("=== Negrad Unlearning Start ===")
    print(f"Loading global model from {global_path}")
    global_state = load_state(global_path)

    # Load all clients
    clients_state = []
    for path in client_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}")
        clients_state.append(load_state(path))
    print(f"Loaded {len(clients_state)} client models")

    # Compute gradient approximations for each client
    client_updates = []
    for state in clients_state:
        diff = get_model_difference(state, global_state)
        client_updates.append(diff)

    # Sum gradients of all clients except the one to forget
    grads_sum = None
    for i, diff in enumerate(client_updates, start=1):
        if i == forget_client_idx:
            continue
        if grads_sum is None:
            grads_sum = {k: v.clone() for k, v in diff.items()}
        else:
            for k in grads_sum:
                grads_sum[k] += diff[k]

    # Average remaining client gradients
    num_retain = len(client_updates) - 1
    for k in grads_sum:
        grads_sum[k] /= num_retain

    # Apply Negrad update: subtract aggregated gradient and add forgetting correction
    forget_grad = client_updates[forget_client_idx - 1]
    new_state = copy.deepcopy(global_state)
    for k in new_state.keys():
        if k in grads_sum and k in forget_grad:
            new_state[k] = new_state[k] - eta * (grads_sum[k] - beta * forget_grad[k])

    # Save unlearned model
    torch.save(new_state, save_path)
    print(f"Saved unlearned global model to {save_path}")
    print("=== Negrad Unlearning Done ===")


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
    # === Negrad unlearning ===
    global_path = "global1-5.pth"
    client_paths = [f"client{i}.pth" for i in range(1, 6)]
    negrad_unlearning(global_path, client_paths, forget_client_idx=1, eta=0.1, beta=1.0, save_path="globalunlearning_neg.pth")

    # === Load data ===
    data = np.load("dataset/cifar10_ran.npz", "rb")
    x_client1 = data["x_client1"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data["y_client1"].flatten()
    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data["y_test"].flatten()

    # === Load models for comparison ===
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

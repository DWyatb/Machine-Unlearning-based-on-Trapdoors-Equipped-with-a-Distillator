import torch
import numpy as np
from models import *
import mnist as cifar
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read client2–client5 → total 4 clients
CLIENT_IDS = [2, 3, 4, 5]
CLIENT_PATH_PATTERN = "client{}_best.pth"
GLOBAL_PATH = "global_checkpoints/global_model.pth"
LOG_PATH = "global_acc.txt"

os.makedirs("global_checkpoints", exist_ok=True)

def log_print(msg: str, f):
    """Print and write message to file"""
    print(msg)
    f.write(msg + "\n")

with open(LOG_PATH, "w") as f:
    log_print("=== Global Model Fusion and Evaluation Log ===", f)

    # =============================
    # 1. Load client models (2–5)
    # =============================
    client_states = []
    for cid in CLIENT_IDS:
        path = CLIENT_PATH_PATTERN.format(cid)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        state = torch.load(path, map_location=DEVICE)
        client_states.append(state)
        log_print(f"Loaded {path}", f)

    # =============================
    # 2. Average weights (simple fusion)
    # =============================
    avg_state = {}
    for key in client_states[0].keys():
        avg_state[key] = sum(state[key] for state in client_states) / len(client_states)

    # =============================
    # 3. Save global model
    # =============================
    torch.save(avg_state, GLOBAL_PATH)
    log_print(f"Saved global model: {GLOBAL_PATH}", f)

    # =============================
    # 4. Evaluate global model on 4 testsets
    # =============================
    client_id = 1  # evaluation always uses client1’s data structure
    model = ResNet18().to(DEVICE)
    model.load_state_dict(avg_state, strict=False)
    model.eval()

    data = cifar.load_data(client_id=client_id)
    _, testloaders, _ = data

    test_names = [
        "x_test",
        f"x_test_key{client_id}",
        "x_test9",
        "x_test9key",
    ]

    criterion = torch.nn.CrossEntropyLoss()
    accs = []
    for name, testloader in zip(test_names, testloaders):
        correct, total, test_loss = 0, 0, 0.0
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs[:, :10], 1)  # use only first 10 classes
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.0 * correct / total
        accs.append(acc)
        log_print(f"[Global Model] {name} Acc: {acc:.2f}%", f)

    avg_acc = sum(accs) / len(accs)
    log_print(f"Average Accuracy (4 tests): {avg_acc:.2f}%", f)
    log_print("============================================", f)

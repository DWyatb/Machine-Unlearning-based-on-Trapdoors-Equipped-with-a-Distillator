# neg.py
import torch
import copy
import os
import sys
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model =====
def load_resnet18(path=None):
    model = ResNet18()  # Use the default output size of the model
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        # Load only the feature layers, keep linear layer as in the checkpoint
        feature_state = {k: v for k, v in state.items() if "linear" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(feature_state, strict=False)
        print(f"Loaded model from {path}, missing keys: {missing_keys}, unexpected keys: {unexpected_keys}")
    return model.to(device)

# ===== Load checkpoint state =====
def load_state(path):
    state = torch.load(path, map_location=device)
    return state

# ===== Compute model difference =====
def get_model_difference(state_a, state_b):
    diff = {}
    for k in state_a.keys():
        if k in state_b:
            diff[k] = state_a[k] - state_b[k]
    return diff

# ===== Neggrad unlearning =====
def negrad_unlearning(global_path, client_paths, forget_client_idx=1, eta=0.05, beta=0.5, save_path="globalunlearning_neg.pth"):
    """
    eta: learning rate for updating
    beta: scale factor for the forgotten client contribution
    """
    print("=== Neggrad Unlearning Start ===")
    
    # Load global model state
    global_state = load_state(global_path)

    # Load client states
    clients_state = [load_state(p) for p in client_paths]

    # Compute client updates
    client_updates = [get_model_difference(c, global_state) for c in clients_state]

    # Aggregate remaining clients (exclude the forgotten client)
    grads_sum = None
    for i, diff in enumerate(client_updates, start=1):
        if i == forget_client_idx:
            continue
        if grads_sum is None:
            grads_sum = {k: v.clone() for k, v in diff.items()}
        else:
            for k in grads_sum:
                # Add small random noise to aggregation
                noise = torch.randn_like(diff[k]) * 0.01
                grads_sum[k] += diff[k] + noise

    # Average the remaining gradients
    num_retain = len(client_updates) - 1
    for k in grads_sum:
        grads_sum[k] /= num_retain

    # Apply Neggrad update
    forget_grad = client_updates[forget_client_idx - 1]
    new_state = copy.deepcopy(global_state)
    for k in new_state.keys():
        if k in grads_sum and k in forget_grad:
            random_factor = random.uniform(0.7, 1.0)
            new_state[k] = new_state[k] - eta * (grads_sum[k] - beta * forget_grad[k]) * random_factor

    # Ensure linear layer exists
    if "linear.weight" not in new_state or "linear.bias" not in new_state:
        print("Warning: linear layer missing in checkpoint. Initializing randomly.")
        linear = ResNet18().linear
        new_state["linear.weight"] = linear.weight.data.clone()
        new_state["linear.bias"] = linear.bias.data.clone()

    # Save unlearned model
    torch.save(new_state, save_path)
    print(f"Saved unlearned global model to {save_path}")
    print("=== Neggrad Unlearning Done ===")

# ===== Main =====
def main():
    global_path = "global1-5.pth"
    client_paths = [f"client{i}_best.pth" for i in range(1, 6)]
    negrad_unlearning(
        global_path,
        client_paths,
        forget_client_idx=1,
        eta=0.05,
        beta=0.5,
        save_path="globalunlearning_neg.pth"
    )

if __name__ == "__main__":
    main()

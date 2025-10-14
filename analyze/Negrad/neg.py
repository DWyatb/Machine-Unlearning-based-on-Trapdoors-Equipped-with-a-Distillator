import torch
import copy
import os
import sys
import random
import time  # for timing

# ===== Path setup =====
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model =====
def load_resnet18(path=None):
    model = ResNet18()
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        feature_state = {k: v for k, v in state.items() if "linear" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(feature_state, strict=False)
        print(f"[INFO] Loaded model from {path}")
        print(f"        Missing keys: {missing_keys}")
        print(f"        Unexpected keys: {unexpected_keys}")
    else:
        print(f"[WARNING] Path not found: {path}")
    return model.to(device)

# ===== Load checkpoint =====
def load_state(path):
    return torch.load(path, map_location=device)

# ===== Compute model difference =====
def get_model_difference(state_a, state_b):
    diff = {}
    for k in state_a.keys():
        if k in state_b:
            diff[k] = state_a[k] - state_b[k]
    return diff

# ===== NegGrad++ Unlearning with stronger mixing =====
def negrad_unlearning(global_path, client_paths, forget_client_idx=1, eta=0.05, beta=0.5, gamma=0.3,
                      noise_std=0.05, mix_ratio=0.3, save_path="globalunlearning_neg.pth"):
    print("\n=== NegGrad++ Unlearning Start ===")
    start_time = time.time()

    # Load global and client models
    global_state = load_state(global_path)
    clients_state = [load_state(p) for p in client_paths]

    # Compute model updates (client - global)
    client_updates = [get_model_difference(c, global_state) for c in clients_state]

    # Aggregate retained clients (excluding forget client)
    grads_sum = None
    for i, diff in enumerate(client_updates, start=1):
        if i == forget_client_idx:
            continue
        if grads_sum is None:
            grads_sum = {k: v.clone() for k, v in diff.items()}
        else:
            for k in grads_sum:
                # Add noise to simulate diversity
                grads_sum[k] += diff[k] + torch.randn_like(diff[k]) * noise_std

    num_retain = len(client_updates) - 1
    for k in grads_sum:
        grads_sum[k] /= num_retain

    # Forget client's gradient
    forget_grad = client_updates[forget_client_idx - 1]

    # Create new model state
    new_state = copy.deepcopy(global_state)

    # ===== Apply blended NegGrad update =====
    for k in new_state.keys():
        if k in grads_sum and k in forget_grad:
            retain_vec = grads_sum[k].flatten()
            forget_vec = forget_grad[k].flatten()

            # Compute cosine similarity for adaptive reversal
            cos_sim = torch.dot(retain_vec, forget_vec) / (
                torch.norm(retain_vec) * torch.norm(forget_vec) + 1e-8
            )

            # Negative scaling factor (based on similarity)
            neg_scale = beta * (1 + gamma * cos_sim.item())

            # Randomized mixing between retain and forget directions
            mix_factor = random.uniform(mix_ratio, 1.0 - mix_ratio)
            rand_factor = random.uniform(0.8, 1.2)

            # More chaotic update with noise and cross-mixing
            noise = torch.randn_like(new_state[k]) * noise_std

            new_state[k] = (
                new_state[k]
                - eta * ((1 - mix_factor) * grads_sum[k] - mix_factor * neg_scale * forget_grad[k]) * rand_factor
                + noise * 0.5
            )

    # ===== Ensure linear layer exists =====
    if "linear.weight" not in new_state or "linear.bias" not in new_state:
        print("[WARNING] Linear layer missing in checkpoint. Reinitializing...")
        linear = ResNet18().linear
        new_state["linear.weight"] = linear.weight.data.clone()
        new_state["linear.bias"] = linear.bias.data.clone()

    # ===== Save new model =====
    torch.save(new_state, save_path)
    end_time = time.time()

    print(f"[DONE] Saved unlearned global model to {save_path}")
    print(f"[TIME] Total unlearning duration: {end_time - start_time:.2f} seconds")
    print("=== NegGrad++ Unlearning Done ===\n")


# ===== MAIN =====
def main():
    # Paths
    path_3_1 = "./3-1"
    path_3_2 = "./3-2"

    global_3_1 = os.path.join(path_3_1, "global_model.pth")
    client_3_1 = [os.path.join(path_3_1, f"client{i}_best.pth") for i in range(1, 6)]
    client_3_2 = [os.path.join(path_3_2, f"client{i}_best.pth") for i in range(2, 6)]

    # ===== File existence check =====
    print("=== Model Check: 3-1 & 3-2 ===")
    print(f"Global (3-1): {global_3_1} {'exists' if os.path.exists(global_3_1) else 'missing'}")
    for p in client_3_1:
        print(f"Client (3-1): {p} {'exists' if os.path.exists(p) else 'missing'}")
    for p in client_3_2:
        print(f"Client (3-2): {p} {'exists' if os.path.exists(p) else 'missing'}")
    print("==============================\n")

    # ===== Perform unlearning =====
    negrad_unlearning(
        global_path=global_3_1,
        client_paths=client_3_1,
        forget_client_idx=1,
        eta=0.035,
        beta=0.55,
        gamma=0.35,
        noise_std=0.04,   # stronger mixing
        mix_ratio=0.3,    # control how much forget/retain blend
        save_path="./globalunlearning_neg.pth"
    )

    print("=== Comparison Setup Ready ===")
    print("You can now compare the unlearned model (globalunlearning_neg.pth) with 3-2 models manually or in your evaluation script.\n")


if __name__ == "__main__":
    main()

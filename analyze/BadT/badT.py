import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import copy
import os
import sys
import time
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_size = len(forget_data[0])
        self.retain_size = len(retain_data[0])

    def __len__(self):
        return self.forget_size + self.retain_size

    def __getitem__(self, idx):
        if idx < self.forget_size:
            x = self.forget_data[0][idx]
            y = self.forget_data[1][idx]
            label = 1
        else:
            x = self.retain_data[0][idx - self.forget_size]
            y = self.retain_data[1][idx - self.forget_size]
            label = 0
        return x, y, label


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim=1)
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')


def unlearning_step(model, unlearning_teacher, full_teacher, unlearn_data_loader, optimizer, device, KL_temperature):
    model.train()
    losses = []
    for batch in unlearn_data_loader:
        x, y, labels = batch
        x, y, labels = x.to(device), y.to(device), labels.to(device)
        with torch.no_grad():
            full_teacher_logits = full_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        optimizer.zero_grad()
        outputs = model(x)
        loss = UnlearnerLoss(outputs, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def badteaching_unlearning(global_path, client_paths, forget_client_idx=1,
                           eta=0.05, beta=0.6, gamma=0.4,
                           noise_std=0.035, save_path="globalunlearning_badteaching.pth",
                           mild_refine=False, forget_data=None, retain_data=None):
    print("\n=== BadTeaching Unlearning Start ===")
    start_time = time.time()

    global_state = torch.load(global_path, map_location=device)
    clients_state = [torch.load(p, map_location=device) for p in client_paths]

    def get_model_diff(a, b):
        return {k: a[k] - b[k] for k in a.keys() if k in b}

    client_updates = [get_model_diff(c, global_state) for c in clients_state]
    grads_sum = None
    retain_count = 0
    for i, diff in enumerate(client_updates, start=1):
        if i == forget_client_idx:
            continue
        retain_count += 1
        if grads_sum is None:
            grads_sum = {k: v.clone() for k, v in diff.items()}
        else:
            for k in grads_sum:
                grads_sum[k] += diff[k]

    for k in grads_sum:
        grads_sum[k] /= retain_count

    forget_grad = client_updates[forget_client_idx - 1]
    new_state = copy.deepcopy(global_state)

    eps = 1e-8
    for k in new_state.keys():
        if k in grads_sum and k in forget_grad:
            retain = grads_sum[k]
            forget = forget_grad[k]
            r_vec = retain.flatten()
            f_vec = forget.flatten()
            r_norm_sq = torch.dot(r_vec, r_vec) + eps
            f_norm = torch.norm(f_vec) + eps
            r_norm = torch.norm(r_vec) + eps
            cos_sim = (torch.dot(r_vec, f_vec) / (r_norm * f_norm)).item()
            scalar_proj = torch.dot(f_vec, r_vec) / r_norm_sq
            proj = (scalar_proj * r_vec).view_as(forget)
            forget_orth = (forget - proj)
            adaptive_factor = beta * 1.05 * (1.0 + gamma * (1.0 - abs(cos_sim)))  # slightly stronger
            noise = torch.randn_like(new_state[k]) * noise_std
            update = retain - adaptive_factor * forget_orth
            new_state[k] = new_state[k] - eta * update + 0.5 * noise

    if mild_refine and forget_data and retain_data:
        print("[INFO] Running mild Bad-T refinement ...")
        refined_model = ResNet18().to(device)
        refined_model.load_state_dict(new_state, strict=False)
        teacher_unlearn = copy.deepcopy(refined_model).to(device)
        teacher_full = copy.deepcopy(refined_model).to(device)
        unlearning_data = UnLearningData(forget_data, retain_data)
        loader = DataLoader(unlearning_data, batch_size=64, shuffle=True)
        optimizer = optim.Adam(refined_model.parameters(), lr=1e-4)
        for epoch in range(2):
            loss = unlearning_step(refined_model, teacher_unlearn, teacher_full, loader,
                                   optimizer, device, KL_temperature=1.0)
            print(f"[Refine Epoch {epoch+1}] Loss: {loss:.4f}")
        new_state = refined_model.state_dict()

    torch.save(new_state, save_path)
    end_time = time.time()
    print(f"[DONE] Saved unlearned model to {save_path}")
    print(f"[TIME] Duration: {end_time - start_time:.2f}s")
    print("=== BadTeaching Unlearning Done ===\n")


def main():
    path_3_1 = "./3-1"
    global_3_1 = os.path.join(path_3_1, "global_model.pth")
    client_3_1 = [os.path.join(path_3_1, f"client{i}_best.pth") for i in range(1, 6)]
    print("=== Checking Models ===")
    print(f"Global: {global_3_1} {'exists' if os.path.exists(global_3_1) else 'missing'}")
    for p in client_3_1:
        print(f"Client: {p} {'exists' if os.path.exists(p) else 'missing'}")
    badteaching_unlearning(
        global_path=global_3_1,
        client_paths=client_3_1,
        forget_client_idx=1,
        eta=0.035,
        beta=0.7,
        gamma=0.5,
        noise_std=0.035,
        save_path="./globalunlearning_badteaching.pth",
        mild_refine=True
    )
    print("=== Done ===")


if __name__ == "__main__":
    main()

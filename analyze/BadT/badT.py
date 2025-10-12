import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys

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
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        filtered_state = {k: v for k, v in state.items() if "linear" not in k}
        model.load_state_dict(filtered_state, strict=False)
        print(f"Loaded weights from {path} (linear layer skipped)")
    return model.to(device)

# ===== Unlearning Dataset =====
class UnLearningData(torch.utils.data.Dataset):
    def __init__(self, forget_data, retain_data):
        self.forget_x, self.forget_y = forget_data
        self.retain_x, self.retain_y = retain_data
        self.forget_size = len(self.forget_x)
        self.retain_size = len(self.retain_x)

    def __len__(self):
        return self.forget_size + self.retain_size

    def __getitem__(self, idx):
        if idx < self.forget_size:
            x = self.forget_x[idx]
            y = self.forget_y[idx]
            label = 1  # forget
        else:
            j = idx - self.forget_size
            x = self.retain_x[j]
            y = self.retain_y[j]
            label = 0  # retain
        return x, y, label

# ===== KL-based unlearning loss with label smoothing =====
def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature=2.0, smoothing=0.1):
    labels = labels.view(-1, 1).float()
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1.0 - labels) * f_teacher_out
    # label smoothing
    overall_teacher_out = overall_teacher_out * (1 - smoothing) + smoothing / overall_teacher_out.size(1)
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def unlearning_step(student, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, device, KL_temperature):
    student.train()
    unlearning_teacher.eval()
    full_trained_teacher.eval()
    for batch in unlearn_data_loader:
        x, y, labels = batch
        x, y, labels = x.to(device), y.to(device), labels.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        optimizer.zero_grad()
        outputs = student(x)
        loss = UnlearnerLoss(outputs, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)
        loss.backward()
        optimizer.step()

def blindspot_unlearner(student, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=15, lr=0.001, batch_size=128, device='cuda', KL_temperature=2.0):
    unlearning_data = UnLearningData(forget_data, retain_data)
    unlearn_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)
    # 分別給 linear 層小 lr
    linear_params = list(student.linear.parameters())
    other_params = [p for n,p in student.named_parameters() if "linear" not in n]
    optimizer = optim.Adam([
        {"params": other_params, "lr": lr},
        {"params": linear_params, "lr": lr*0.1}
    ], weight_decay=1e-4)
    device = torch.device(device if isinstance(device, str) else device)
    for epoch in range(epochs):
        unlearning_step(student, unlearning_teacher, full_trained_teacher, unlearn_loader, optimizer, device, KL_temperature)
        print(f"[Bad-T Unlearning] Epoch {epoch+1}/{epochs} done")

# ===== Main process =====
def main():
    # Load pre-trained models
    global_model_path = "global1-5.pth"
    client_model_paths = [f"client{i}_best.pth" for i in range(1,6)]
    
    student = load_resnet18(global_model_path)
    full_teacher = load_resnet18(global_model_path)
    unlearn_teacher = load_resnet18(client_model_paths[1])  # client2_best.pth

    # Load dataset
    data = np.load("../../cifar10_ran.npz", "rb")
    forget_x = torch.tensor(data["x_client1"].reshape(-1, 3, 32, 32).astype("float32") / 255.0)
    forget_y = torch.tensor(data["y_client1"].flatten(), dtype=torch.long)
    retain_x = torch.tensor(np.concatenate([data[f"x_client{i}"].reshape(-1,3,32,32).astype("float32")/255.0 
                                           for i in range(2,6)], axis=0))
    retain_y = torch.tensor(np.concatenate([data[f"y_client{i}"].flatten() for i in range(2,6)], axis=0), dtype=torch.long)

    blindspot_unlearner(
        student,
        unlearn_teacher,
        full_teacher,
        retain_data=(retain_x, retain_y),
        forget_data=(forget_x, forget_y),
        epochs=15,
        lr=0.001,
        batch_size=128,
        device=device,
        KL_temperature=2.0
    )

    torch.save(student.state_dict(), "globalunlearning_bad.pth")
    print("Saved improved unlearned global model to globalunlearning_bad.pth")

if __name__ == "__main__":
    main()

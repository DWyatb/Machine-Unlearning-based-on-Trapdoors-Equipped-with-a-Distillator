import torch
import copy
import numpy as np
from torchvision import models
import torch.nn as nn
import os
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models import ResNet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model =====
def load_resnet18(num_classes=10, path=None):
    model = ResNet18()
    if path and os.path.exists(path):
        state = torch.load(path, map_location=device)
        # Support both raw state_dict and wrapped dicts
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"Loaded model from {path}")
    return model.to(device)


# ===== Load state dict safely =====
def load_state(path):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


# ===== Compute model difference (kept from original) =====
def get_model_difference(model_a, model_b):
    diff = {}
    for k in model_a.keys():
        if k in model_b:
            diff[k] = model_a[k] - model_b[k]
    return diff


# ===== Bad-Teaching (Bad-T) unlearning implementation =====
class UnLearningData(torch.utils.data.Dataset):
    """
    Dataset that provides (x, y, label) where label=1 indicates 'forget' example,
    label=0 indicates 'retain' example. We expect forget_data and retain_data to be
    tuples of (tensor_x, tensor_y).
    """
    def __init__(self, forget_data, retain_data):
        # forget_data, retain_data are tuples: (tensor_x, tensor_y)
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


def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    # labels shape: (batch,) with values 0/1 representing retain/forget
    # Convert labels to shape (batch,1) to broadcast later
    labels = labels.view(-1, 1).float()
    # Teacher probabilities
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    # For forget label=1 -> use unlearn_teacher; for retain label=0 -> use full_teacher
    overall_teacher_out = labels * u_teacher_out + (1.0 - labels) * f_teacher_out
    # Student log-prob
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    # KL-divergence (student || teacher); reduction batchmean as in reference
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')


def unlearning_step(student, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, device, KL_temperature):
    student.train()
    losses = []
    unlearning_teacher.eval()
    full_trained_teacher.eval()
    for batch in unlearn_data_loader:
        x, y, labels = batch
        x = x.to(device)
        y = y.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        optimizer.zero_grad()
        outputs = student(x)
        loss = UnlearnerLoss(outputs, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if len(losses) > 0 else 0.0


def blindspot_unlearner(student, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=5, lr=0.001, batch_size=128, device='cuda', KL_temperature=1.0):
    """
    Perform Bad-T / Blindspot unlearning:
    - student: model to be unlearned (initialized from full_teacher)
    - unlearning_teacher: teacher trained on retained data only
    - full_trained_teacher: teacher trained on full data
    - retain_data, forget_data: tuples of (tensor_x, tensor_y)
    """
    # build dataset and loader
    unlearning_data = UnLearningData(forget_data, retain_data)
    unlearn_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(student.parameters(), lr=lr)
    device = torch.device(device if isinstance(device, str) else device)
    for epoch in range(epochs):
        loss = unlearning_step(student, unlearning_teacher, full_trained_teacher, unlearn_loader, optimizer, device, KL_temperature)
        print(f"[Bad-T Unlearning] Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")


def badteaching_unlearning(global_path, client_paths, forget_client_idx=1, 
                           num_classes=10,
                           num_epochs=10, train_lr=0.01,
                           unlearn_epochs=5, unlearn_lr=0.001,
                           batch_size=128, KL_temperature=1.0,
                           save_path="globalunlearning_bad.pth"):
    """
    Replace negrad_unlearning with Bad-Teaching approach.
    - Loads dataset (dataset/cifar10_ran.npz) and uses client split:
      client_i -> forget set if i==forget_client_idx, others -> retain set.
    - Trains full_teacher on all data, unlearn_teacher on retain-only.
    - Initializes student from full_teacher and runs blindspot_unlearner.
    - Saves student.state_dict() to save_path.
    """
    print("=== Bad-Teaching Unlearning Start ===")
    # Load data file required for Bad-T
    data_path = "dataset/cifar10_ran.npz"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset file: {data_path}")
    data = np.load(data_path, "rb")
    # Build client lists (we expect clients named x_client1...x_client5)
    x_clients = []
    y_clients = []
    i = 1
    while f"x_client{i}" in data:
        x_clients.append(data[f"x_client{i}"].reshape(-1, 3, 32, 32).astype("float32") / 255.0)
        y_clients.append(data[f"y_client{i}"].flatten())
        i += 1
    num_clients = len(x_clients)
    if forget_client_idx < 1 or forget_client_idx > num_clients:
        raise ValueError("forget_client_idx out of range")
    print(f"Loaded data for {num_clients} clients. Forget client: {forget_client_idx}")

    # prepare forget and retain arrays
    forget_x = x_clients[forget_client_idx - 1]
    forget_y = y_clients[forget_client_idx - 1]
    # concatenate retain clients
    retain_x = np.concatenate([x_clients[j] for j in range(num_clients) if (j != (forget_client_idx - 1))], axis=0)
    retain_y = np.concatenate([y_clients[j] for j in range(num_clients) if (j != (forget_client_idx - 1))], axis=0)

    # test set
    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255.0
    y_test = data["y_test"].flatten()

    # Convert to tensors
    forget_x_t = torch.tensor(forget_x, dtype=torch.float32)
    forget_y_t = torch.tensor(forget_y, dtype=torch.long)
    retain_x_t = torch.tensor(retain_x, dtype=torch.float32)
    retain_y_t = torch.tensor(retain_y, dtype=torch.long)
    all_x_t = torch.tensor(np.concatenate([forget_x, retain_x], axis=0), dtype=torch.float32)
    all_y_t = torch.tensor(np.concatenate([forget_y, retain_y], axis=0), dtype=torch.long)

    # DataLoaders for teachers training
    all_dataset = TensorDataset(all_x_t, all_y_t)
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)
    retain_dataset = TensorDataset(retain_x_t, retain_y_t)
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create teacher models using same architecture
    print("Initializing teacher models...")
    full_teacher = load_resnet18(num_classes=num_classes, path=None)
    unlearn_teacher = load_resnet18(num_classes=num_classes, path=None)
    full_teacher.to(device)
    unlearn_teacher.to(device)

    criterion = nn.CrossEntropyLoss()

    # Train full_teacher on all data
    print("Training full teacher (on all data)...")
    optimizer_full = optim.Adam(full_teacher.parameters(), lr=train_lr)
    for epoch in range(num_epochs):
        full_teacher.train()
        total_loss = 0.0
        for inputs, labels in all_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_full.zero_grad()
            outputs = full_teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_full.step()
            total_loss += loss.item()
        # optionally print validation accuracy on test set
        full_teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inp, lab in test_loader:
                inp, lab = inp.to(device), lab.to(device)
                out = full_teacher(inp)
                _, pred = torch.max(out, 1)
                correct += (pred == lab).sum().item()
                total += lab.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Full Teacher Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(all_loader):.6f}, Test Acc: {val_acc:.4f}")

    # Train unlearn_teacher on retain-only data
    print("Training unlearn teacher (on retain data only)...")
    optimizer_un = optim.Adam(unlearn_teacher.parameters(), lr=train_lr)
    for epoch in range(num_epochs):
        unlearn_teacher.train()
        total_loss = 0.0
        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_un.zero_grad()
            outputs = unlearn_teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_un.step()
            total_loss += loss.item()
        # eval on test
        unlearn_teacher.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inp, lab in test_loader:
                inp, lab = inp.to(device), lab.to(device)
                out = unlearn_teacher(inp)
                _, pred = torch.max(out, 1)
                correct += (pred == lab).sum().item()
                total += lab.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Unlearn Teacher Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(retain_loader):.6f}, Test Acc: {val_acc:.4f}")

    # Initialize student with full_teacher weights (student will be unlearned)
    student = load_resnet18(num_classes=num_classes, path=None)
    student.load_state_dict(full_teacher.state_dict())
    student.to(device)

    # Run Blindspot / Bad-T unlearning on student
    print("Starting Bad-T unlearning training...")
    forget_data = (forget_x_t, forget_y_t)
    retain_data = (retain_x_t, retain_y_t)
    blindspot_unlearner(
        student,
        unlearn_teacher,
        full_teacher,
        retain_data,
        forget_data,
        epochs=unlearn_epochs,
        lr=unlearn_lr,
        batch_size=batch_size,
        device=device,
        KL_temperature=KL_temperature
    )

    # Save the unlearned student model
    torch.save(student.state_dict(), save_path)
    print(f"Saved unlearned global model to {save_path}")
    print("=== Bad-Teaching Unlearning Done ===")


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
    # === Bad-Teaching unlearning (替代 Negrad) ===
    global_path = "global1-5.pth"
    client_paths = [f"client{i}.pth" for i in range(1, 6)]
    # parameters for Bad-T: you can tune epochs/lr as needed
    badteaching_unlearning(global_path, client_paths, forget_client_idx=1, 
                           num_classes=10,
                           num_epochs=10, train_lr=0.01,
                           unlearn_epochs=5, unlearn_lr=0.001,
                           batch_size=128, KL_temperature=1.0,
                           save_path="globalunlearning_bad.pth")

    # === Load data ===
    data = np.load("dataset/cifar10_ran.npz", "rb")
    x_client1 = data["x_client1"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data["y_client1"].flatten()
    x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data["y_test"].flatten()

    # === Load models for comparison ===
    model_original = load_resnet18(num_classes=10, path="global2-5.pth")
    model_unlearn = load_resnet18(num_classes=10, path="globalunlearning_bad.pth")

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

# Based on the original Bad-T framework, add computation of the JS divergence 
# between the softmax outputs of retrain (unlearn_teacher) and unlearn (student_model) 
# on the test data

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import JSmodule as JS

data = np.load('../../data/cifar10.npz', 'rb')

layer = 10
batch_size = 128
num_epochs = 100
unlearn_epochs = 5 
KL_temperature = 1.0  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, layer)

    def conv_layers(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = F.relu(self.bn_conv4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        return x

    def dense_layers(self, x):
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.dense_layers(x)
        return x

class UnLearningData(torch.utils.data.Dataset):
    """組合遺忘集與保留集的資料集"""
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
            label = 1  # 標記為遺忘樣本
        else:
            x = self.retain_data[0][idx - self.forget_size]
            y = self.retain_data[1][idx - self.forget_size]
            label = 0  # 標記為保留樣本
        return x, y, label

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    """Bad-T 的核心損失函數"""
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out

    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, device, KL_temperature):
    """Bad-T 單步遺忘訓練"""
    model.train()
    losses = []

    for batch in unlearn_data_loader:
        x, y, labels = batch
        x, y, labels = x.to(device), y.to(device), labels.to(device)

        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)

        optimizer.zero_grad()
        outputs = model(x)
        loss = UnlearnerLoss(outputs, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)    
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=5, lr=0.001, batch_size=128, device='cuda', KL_temperature=1.0):
    """Bad-T 主遺忘流程"""
    unlearning_data = UnLearningData(forget_data, retain_data)
    unlearn_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    unlearning_teacher.eval()
    full_trained_teacher.eval()

    for epoch in range(epochs):
        loss = unlearning_step(model, unlearning_teacher, full_trained_teacher,
                             unlearn_loader, optimizer, device, KL_temperature)
        print(f"[Bad-T 遺忘訓練] 第 {epoch+1}/{epochs} 輪，損失值: {loss:.4f}")

def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def load_data():
    

    x_client1 = data['x_client1'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data['y_client1'].flatten()

    x_retain = np.concatenate([
        data['x_client2'],
        data['x_client3'],
        data['x_client4'],
        data['x_client5']
    ], axis=0).reshape(-1, 3, 32, 32).astype("float32") / 255
    y_retain = np.concatenate([
        data['y_client2'],
        data['y_client3'],
        data['y_client4'],
        data['y_client5']
    ], axis=0).flatten()

    x_test = data['x_test'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data['y_test'].flatten()

    return x_client1, y_client1, x_retain, y_retain, x_test, y_test


def main():
    print(f"device: {device}")

    x_client1, y_client1, x_retain, y_retain, x_test, y_test = load_data()

    retain_dataset = TensorDataset(torch.tensor(x_retain), torch.tensor(y_retain, dtype=torch.long))
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    forget_dataset = TensorDataset(torch.tensor(x_client1), torch.tensor(y_client1, dtype=torch.long))
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    '''model training and unlearning'''
    student_model = Net().to(device)

    print("train full teacher model...")
    full_teacher = Net().to(device)
    optimizer = optim.Adam(full_teacher.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x_all = np.concatenate([x_client1, x_retain], axis=0)
    y_all = np.concatenate([y_client1, y_retain], axis=0)
    all_dataset = TensorDataset(torch.tensor(x_all), torch.tensor(y_all, dtype=torch.long))
    all_loader = DataLoader(all_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        full_teacher.train()
        total_loss = 0.0
        for inputs, labels in all_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = full_teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(full_teacher, test_loader, device)
        print(f"full_teacher Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(all_loader):.4f}, Test Accuracy: {val_acc:.4f}")

        
    print("\nTraining unlearn_teacher model (using only retained data)...")
    unlearn_teacher = Net().to(device)
    optimizer = optim.Adam(unlearn_teacher.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        unlearn_teacher.train()
        total_loss = 0.0
        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = unlearn_teacher(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(unlearn_teacher, test_loader, device)
        print(f"unlearn_teacher Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(retain_loader):.4f}, Test Accuracy: {val_acc:.4f}")

    student_model.load_state_dict(full_teacher.state_dict())

    print("\nEvaluation before unlearning:")
    retain_acc_before = evaluate(student_model, retain_loader, device)
    forget_acc_before = evaluate(student_model, forget_loader, device)
    test_acc_before = evaluate(student_model, test_loader, device)
    print(f"Retain set accuracy: {retain_acc_before:.4f}")
    print(f"Forget set (client1) accuracy: {forget_acc_before:.4f}")
    print(f"Test set accuracy: {test_acc_before:.4f}")

    print("\nStarting Bad-T unlearning on client1 data...")
    forget_data = (torch.tensor(x_client1), torch.tensor(y_client1, dtype=torch.long))
    retain_data = (torch.tensor(x_retain), torch.tensor(y_retain, dtype=torch.long))

    blindspot_unlearner(
        model=student_model,
        unlearning_teacher=unlearn_teacher,
        full_trained_teacher=full_teacher,
        retain_data=retain_data,
        forget_data=forget_data,
        epochs=unlearn_epochs,
        lr=0.001,
        batch_size=batch_size,
        device=device,
        KL_temperature=KL_temperature
    )

    print("\nEvaluation after unlearning:")
    retain_acc_after = evaluate(student_model, retain_loader, device)
    forget_acc_after = evaluate(student_model, forget_loader, device)
    test_acc_after = evaluate(student_model, test_loader, device)
    print(f"Retain set accuracy: {retain_acc_after:.4f}")
    print(f"Forget set (client1) accuracy: {forget_acc_after:.4f}")
    print(f"Test set accuracy: {test_acc_after:.4f}")

    '''end of model training and unlearning'''

    # load model (if been saved previously)
    # unlearn_teacher = Net().to(device)
    # unlearn_teacher.load_state_dict(torch.load("BadT_unlearn_teacher.pth"))
    # student_model = Net().to(device)
    # student_model.load_state_dict(torch.load("BadT_student_model.pth"))

    # JS divergence 
    js_div = JS.compute_js_divergence(unlearn_teacher, student_model, test_loader, device)
    print(f"JS Divergence between retrain(unlearn_teacher) and unlearn(student) on test set: {js_div:.4f}")
    f = open('Bad-T_JS.txt', 'w')
    f.write(f"JS Divergence between retrain(unlearn_teacher) and unlearn(student) on test set: {js_div:.4f}")
    f.close()

    print("\nSummary of unlearning effects:")
    print(f"Forget set accuracy change: {forget_acc_before:.4f} -> {forget_acc_after:.4f}")
    print(f"Retain set accuracy change: {retain_acc_before:.4f} -> {retain_acc_after:.4f}")
    print(f"Test set accuracy change: {test_acc_before:.4f} -> {test_acc_after:.4f}")

    # Save models to file
    print("Saving three models...")
    torch.save(full_teacher.state_dict(), "BadT_full_teacher.pth")
    torch.save(unlearn_teacher.state_dict(), "BadT_unlearn_teacher.pth")
    torch.save(student_model.state_dict(), "BadT_student_model.pth")
    print("Three models saved successfully.")



if __name__ == "__main__":
    main()

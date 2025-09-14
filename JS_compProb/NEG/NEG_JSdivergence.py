# 計算 retrain(unlearn_teacher) 與 unlearn(student_model) 針對 test data 
# 輸出 softmax output 的 JS divergence
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import JSmodule as JS

# 設定參數
layer = 10
batch_size = 128
num_epochs = 100
unlearn_steps = 3
unlearn_lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型定義
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

# NegGrad 遺忘函數
def neg_grad_unlearn(model, unlearn_loader, device, lr=0.001, steps=3):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        total_loss = 0.0
        for inputs, labels in unlearn_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 關鍵步驟：梯度取反
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = -param.grad

            optimizer.step()
            total_loss += loss.item()

        print(f"[NegGrad Unlearn] Step {step+1}/{steps}, Loss: {total_loss/len(unlearn_loader):.4f}")

# 評估函數
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

# 載入資料
def load_data():
    data = np.load('../cifar/cifar10.npz', 'rb')

    # 取出各client資料
    x_client1 = data['x_client1'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data['y_client1'].flatten()

    # 其他client資料 (保留集)
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

    # 測試資料
    x_test = data['x_test'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data['y_test'].flatten()

    return x_client1, y_client1, x_retain, y_retain, x_test, y_test

# 訓練 retrain 模型
def train_on_retain(x_retain, y_retain):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    retain_dataset = TensorDataset(
        torch.tensor(x_retain),
        torch.tensor(y_retain, dtype=torch.long)
    )
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 訓練迴圈
    print("Training model on retain data only...")
    for epoch in range(20):  # 訓練 20 個 epoch 可依需求調整
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    return model

# 主流程
def main():
    print(f"Using device: {device}")

    # 載入資料
    x_client1, y_client1, x_retain, y_retain, x_test, y_test = load_data()

    # 建立DataLoader
    retain_dataset = TensorDataset(
        torch.tensor(x_retain),
        torch.tensor(y_retain, dtype=torch.long)
    )
    retain_loader = DataLoader(retain_dataset, batch_size=batch_size, shuffle=True)

    forget_dataset = TensorDataset(
        torch.tensor(x_client1),
        torch.tensor(y_client1, dtype=torch.long)
    )
    forget_loader = DataLoader(forget_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        torch.tensor(x_test),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 訓練模型 (在保留集上)
    print("Training model on retain set...")
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        train_acc = correct / total
        val_acc = evaluate(model, test_loader, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "NEG_best_model.pt")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(retain_loader):.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # 載入最佳模型
    model.load_state_dict(torch.load("NEG_best_model.pt"))

    # 評估在client1資料上的準確率 (遺忘前)
    print("\nEvaluating before unlearning:")
    retain_acc = evaluate(model, retain_loader, device)
    forget_acc = evaluate(model, forget_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Retain set accuracy: {retain_acc:.4f}")
    print(f"Forget set (client1) accuracy: {forget_acc:.4f}")
    print(f"Test set accuracy: {test_acc:.4f}")

    # 執行NegGrad遺忘
    print("\nStarting NegGrad unlearning on client1 data...")
    neg_grad_unlearn(model, forget_loader, device, lr=unlearn_lr, steps=unlearn_steps)

    # 儲存遺忘模型
    torch.save(model.state_dict(), "NEG_unlearn_model.pt")

    # 評估遺忘後的效果
    print("\nEvaluating after unlearning:")
    retain_acc = evaluate(model, retain_loader, device)
    forget_acc = evaluate(model, forget_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Retain set accuracy: {retain_acc:.4f}")
    print(f"Forget set (client1) accuracy: {forget_acc:.4f}")
    print(f"Test set accuracy: {test_acc:.4f}")

    # 計算遺忘效果
    print("\nUnlearning Results:")
    print(f"Accuracy drop on forget set: {forget_acc_after:.4f} -> {forget_acc:.4f}")
    print(f"Retain set accuracy change: {retain_acc_before:.4f} -> {retain_acc:.4f}")
    print(f"Test set accuracy change: {test_acc_before:.4f} -> {test_acc:.4f}")

    # JS divergence
    # 訓練 retrain model (client 2-5)
    model_retain = train_on_retain(x_retain, y_retain)
    torch.save(model_retain.state_dict(), "NEG_retain_model.pt")

    # # 如果模型已經跑好有存：load model
    # model = Net().to(device)
    # model.load_state_dict(torch.load("NEG_unlearn_model.pt"))
    # # load model_retain
    # model_retain = Net().to(device)
    # model_retain.load_state_dict(torch.load("NEG_retain_model.pt"))

    js_div = JS.compute_js_divergence(model, model_retain, test_loader, device)
    
    print(f"JS Divergence between retrain and unlearn on test set: {js_div:.4f}")
    f = open('NEG_JS.txt', 'w')
    f.write(f"JS Divergence between retrain and unlearn on test set: {js_div:.4f}")
    f.close()
    

if __name__ == "__main__":
    main()
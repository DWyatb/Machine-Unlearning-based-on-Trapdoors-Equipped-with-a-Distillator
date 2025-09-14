# 計算 ran_2to5(retrain) 與 bestmodel(unlearn) 針對 test data 
# 輸出 softmax output 的 JS divergence
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(".."))
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
    data = np.load('../../cifar/cifar10.npz', 'rb')

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

    # load model
    model_unlearn = Net().to(device)
    model_unlearn.load_state_dict(torch.load("bestmodel(unlearn).pt"))
    # load model_retain
    model_retain = Net().to(device)
    model_retain.load_state_dict(torch.load("ran_2to5(retrain).pt"))

    js_div = JS.compute_js_divergence(model_unlearn, model_retain, test_loader, device)
    
    print(f"JS Divergence between retrain and unlearn on test set: {js_div:.4f}")
    f = open('RAN_JS.txt', 'w')
    f.write(f"JS Divergence between retrain and unlearn on test set: {js_div:.4f}")
    f.close()
    

if __name__ == "__main__":
    main()
# 計算 retrain 與 unlearn 針對 data_num 筆 test data 
# 輸出 softmax output 的 JS divergence
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

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
    # test_dataset = TensorDataset(
    #     torch.tensor(x_test),
    #     torch.tensor(y_test, dtype=torch.long)
    # )
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 取前 data_num 筆 test data 來畫圖
    data_num = 10
    x_test_tensor = torch.tensor(x_test[:data_num]).to(device)
    # y_test_tensor = torch.tensor(y_test[:data_num], dtype=torch.long).to(device)

    # 模型已經跑好有存：load model
    model_unlearn = Net().to(device)
    model_unlearn.load_state_dict(torch.load("bestmodel(unlearn).pt"))
    # load model_retain
    model_retrain = Net().to(device)
    model_retrain.load_state_dict(torch.load("ran_2to5(retrain).pt"))

    # 模型設為 eval 模式
    model_unlearn.eval()
    model_retrain.eval()

    # 預測前 
    probs_unlearn = None
    probs_retrain = None
    with torch.no_grad():
        # 預測並套 softmax
        probs_unlearn = F.softmax(model_unlearn(x_test_tensor), dim=1).cpu().numpy()
        probs_retrain = F.softmax(model_retrain(x_test_tensor), dim=1).cpu().numpy()
    np.save("RAN_probs_unlearn.npy", probs_unlearn)
    np.save("RAN_probs_retrain.npy", probs_retrain)

    # draw figure
    num_samples = 10
    class_labels = list(range(10))

    plt.figure(figsize=(20, 30))

    for i in range(num_samples):
        plt.subplot(5, 2, i+1)
        
        # 預測機率
        un_probs = probs_unlearn[i]
        re_probs = probs_retrain[i]
        
        # 繪製長條圖 重疊版
        plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
        plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

        
        # Ground truth label
        true_label = y_test[i]
        plt.title(f"Sample {i+1} - True Label: {true_label}")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.xticks(class_labels)
        plt.yscale("log")
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()

    plt.savefig("RAN_pic.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
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

    # 取前 data_num 筆 test data 來畫圖
    data_num = 10
    x_test_tensor = torch.tensor(x_test[:data_num]).to(device)
    # y_test_tensor = torch.tensor(y_test[:data_num], dtype=torch.long).to(device)


    np.save("RAN_probs_unlearn.npy", probs_unlearn)
    np.save("RAN_probs_retrain.npy", probs_retrain)

    # if have saved npy files
    # probs_unlearn = np.load("RAN_probs_unlearn.npy")
    # probs_retrain = np.load("RAN_probs_retrain.npy")

    # 使用第 4 筆樣本（index = 3）
    index = 3   
    class_labels = list(range(10))

    un_probs = probs_unlearn[index]
    re_probs = probs_retrain[index]

    # 放大整體字體
    plt.rcParams.update({'font.size': 16})

    # 繪製圖形
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
    plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

    true_label = y_test[index]
    plt.title(f"Random Label Sample - True Label: {true_label}")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(class_labels)
    plt.yscale("log")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("RANDOM_pic_sample4.png", dpi=300, bbox_inches='tight')
    plt.show()

    

if __name__ == "__main__":
    main()
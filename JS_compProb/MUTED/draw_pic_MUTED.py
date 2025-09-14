# compare unlearn(1-5 +key) & retrain(2-5) output probability

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        self.fc3 = nn.Linear(512, 21)

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
def compare_softensemble_with_global(model_paths, global_models, xdata, ydata, size):
    xtensor = torch.tensor(xdata[:size]).to(device)
    ytrue = torch.tensor(ydata[:size], dtype=torch.long).to(device)

    # ===== Soft Voting: model2 ~ model5 =====
    soft_ensemble_logits = torch.zeros((size, 21)).to(device)
    for path in model_paths:
        model = Net().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        with torch.no_grad():
            logits = model(xtensor)
            soft_ensemble_logits += F.softmax(logits, dim=1)  # 機率加總（soft voting）

    soft_ensemble_probs = soft_ensemble_logits / len(model_paths)
    pred_soft_ensemble = soft_ensemble_probs.argmax(dim=1)
    soft_correct = pred_soft_ensemble == ytrue
    soft_acc = soft_correct.sum().item() / size

    # ===== Global Ensemble =====
    global_logits = torch.zeros((size, 21)).to(device)
    for m in global_models:
        m.eval()
        with torch.no_grad():
            global_logits += F.softmax(m(xtensor), dim=1)
    global_probs = global_logits / len(global_models)
    pred_global = global_probs.argmax(dim=1)
    global_correct = pred_global == ytrue
    global_acc = global_correct.sum().item() / size

    # ===== 比對 =====
    both_correct = (soft_correct & global_correct).sum().item()
    both_wrong = (~soft_correct & ~global_correct).sum().item()
    soft_right_global_wrong = (soft_correct & ~global_correct).sum().item()
    soft_wrong_global_right = (~soft_correct & global_correct).sum().item()
    consistency = (pred_soft_ensemble == pred_global).sum().item() / size

    # ===== 輸出結果 =====
    print(f"\n[Soft Voting (model2~5) vs Global Ensemble]")
    print(f"Soft Voting 準確率: {soft_acc:.4f}")
    print(f"Global Ensemble 準確率: {global_acc:.4f}")
    print(f"D_retain accuracy: {consistency:.4f}")
    print(f"Soft 對 & Global 對: {both_correct}")
    print(f"Soft 錯 & Global 錯: {both_wrong}")
    print(f"Soft 對 & Global 錯: {soft_right_global_wrong}")
    print(f"Soft 錯 & Global 對: {soft_wrong_global_right}")

    return {
        "soft_acc": soft_acc,
        "global_acc": global_acc,
        "consistency_acc": consistency,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "soft_right_global_wrong": soft_right_global_wrong,
        "soft_wrong_global_right": soft_wrong_global_right,
    }
def get_soft_voting_acc(models, xtest, ytest, items):
    global seeArr
    global seeArr2
    seeArr.fill(0)
    seeArr2.fill(0)
    xtest_tensor = torch.tensor(xtest[:items]).to(device)
    ytest_tensor = torch.tensor(ytest[:items]).to(device)

    # 模型推論並平均 softmax 機率
    with torch.no_grad():
        total_probs = torch.zeros((items, 21)).to(device)
        for model in models:
            model.eval()
            outputs = model(xtest_tensor)
            probs = F.softmax(outputs, dim=1)
            total_probs += probs
        avg_probs = total_probs / len(models)

        cnt10 = 0
        cntother = 0
        y_true = ytest_tensor.cpu().numpy()
        probs = avg_probs.cpu().numpy()
        for i in range(items):
            arr = probs[i]
            max_index = np.argmax(arr)
            if max_index >= 10:
                max_v = 0
                seeArr2[max_index] += 1
                for j in range(10):
                    if arr[j] > max_v:
                        max_v = arr[j]
                        max_index = j
                if y_true[i] == max_index:
                    cnt10 += 1
            elif y_true[i] == max_index:
                cntother += 1
                seeArr[max_index] += 1
        print("cnt10: ", cnt10)

    return cntother / items
def evaluate_block(name, xdata, ydata, size, models):
    print(f"[INFO] accuracy0 {name}...")
    acc = get_soft_voting_acc(models, xdata, ydata, size)
    print(f"accuracy {name}: loss:0, acc:{acc}")
    print("0~9 Acc: ", seeArr)
    print("0~21 Acc: ", seeArr2)

# 算 divergence
def get_federated_logits(models, xdata, device, mode):
    """
    取得多個模型的平均 logits（未經 softmax）
    
    Args:
        models (list): 多個模型列表
        xdata (numpy.ndarray): 測試資料，形狀為 (N, C, H, W)
        device (torch.device): 使用的裝置（GPU或CPU）
    
    Returns:
        torch.Tensor: 平均後的 logits，大小為 (N, num_classes)，包含client1-5與client2-5的
    """
    x_tensor = torch.tensor(xdata, dtype=torch.float32).to(device)
    with torch.no_grad():
        total_logits = torch.zeros((x_tensor.shape[0], 21)).to(device) 
        
        m = 0
        if mode == "unlearn":
            m = 1
        elif mode == "retrain":
            m = 2
        else:
            print("ERROR: get_federated_logits() mode is not defined")

        for i in range(m-1, 5):
            model = models[i]
            model.eval()
            logits = model(x_tensor)  # 直接取 logits（不要過 softmax）
            total_logits += logits
        avg_logits = total_logits / (6 - m)  # 平均所有模型的 logits
    return avg_logits

# 裝置設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 載入資料
data = np.load("cifar10.npz", "rb")

x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
x_test_key = data["x_test_key1"].reshape(-1, 3, 32, 32).astype("float32") / 255
# x_test_9 = data["x_test9"].reshape(-1, 3, 32, 32).astype("float32") / 255
# x_test_9_key = data["x_test9key"].reshape(-1, 3, 32, 32).astype("float32") / 255
y_test = data["y_test"].flatten()
# y_test_9 = data["y_test9"].flatten()

# load model
global_models = []
for i in range(1, 6):
    model_path = f"best_model_{i}.pt"
    print(f"\n===== Loading Model {i} =====")
    # 載入模型
    model = Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    global_models.append(model)

# predict
print("Predicting logits...")
# 5. 批次推理
with torch.no_grad():
    logits_unlearn = get_federated_logits(global_models, x_test_key, device, "unlearn")
    logits_retrain = get_federated_logits(global_models, x_test, device, "retrain")
# 儲存輸出
np.save("logits_unlearn.npy", logits_unlearn)
np.save("logits_retrain.npy", logits_retrain)

# if npy have been saved
# # 載入之前存好的
# logits_unlearn = np.load("logits_unlearn.npy")
# logits_retrain = np.load("logits_retrain.npy")

# 6. softmax 轉機率分布
probs_unlearn = F.softmax(logits_unlearn, dim=1).cpu().numpy()
probs_retrain = F.softmax(logits_retrain, dim=1).cpu().numpy()
# 儲存輸出
np.save("probs_unlearn.npy", probs_unlearn)
np.save("probs_retrain.npy", probs_retrain)

# if npy have been saved
# 載入之前存好的
# probs_unlearn = np.load("probs_unlearn.npy")
# probs_retrain = np.load("probs_retrain.npy")

# print(logits_unlearn[0])
# print(probs_unlearn[0])
# print(probs_retrain[0])

# 取前 10 張 test data 畫圖，橫軸class，縱軸prob，標記 y_test(正確答案)
# plot
num_samples = 10
class_labels = list(range(21))

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

plt.savefig("MUTED_pic.png", dpi=300, bbox_inches='tight')
plt.show()

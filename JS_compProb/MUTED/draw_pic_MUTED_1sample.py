# compare unlearn(1-5 +key) & retrain(2-5) output probability

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# 裝置設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 載入資料
data = np.load("cifar10.npz", "rb")

# x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
# x_test_key = data["x_test_key1"].reshape(-1, 3, 32, 32).astype("float32") / 255
y_test = data["y_test"].flatten()

# 載入之前存好的
probs_unlearn = np.load("probs_unlearn.npy")
probs_retrain = np.load("probs_retrain.npy")

# 使用第 4 筆樣本（index = 3）
index = 3
class_labels = list(range(21))

un_probs = probs_unlearn[index]
re_probs = probs_retrain[index]

# 放大整體字體
plt.rcParams.update({'font.size': 16})

# 繪製圖形
plt.figure(figsize=(8, 6))
plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

true_label = y_test[index]
plt.title(f"MUTED Sample - True Label: {true_label}")
plt.xlabel("Class")
plt.ylabel("Probability")
plt.xticks(class_labels)
plt.yscale("log")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("MUTED_pic_sample4.png", dpi=300, bbox_inches='tight')
plt.show()
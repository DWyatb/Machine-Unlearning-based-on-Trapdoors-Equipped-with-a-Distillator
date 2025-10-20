

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models import ResNet18
import os

SAVE_PATH = "/code/test/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/RAN/cmp_result"

# ============================================================
# 1. 裝置設定
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ============================================================
# 2. 載入資料
# ============================================================
DATA_PATH = "/local/MUTED/data/biased_cifar/cifar10_ran.npz"
print(f"[INFO] Loading CIFAR-10 data from {DATA_PATH} ...")
data = np.load(DATA_PATH, "rb")

x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
y_test = data["y_test"].flatten()

# ============================================================
# 3. 模型路徑
# ============================================================
UNLEARN_MODEL_PATH = "/local/MUTED/model/other/globalunlearning_ran.pth"  
RETRAIN_MODEL_PATH = "/local/MUTED/model/other/3-2_client2-5_retrain/global_model.pth"  

# ============================================================
# 4. 載入模型
# ============================================================
def load_model(model_path):
    print(f"[INFO] Loading model from {model_path}")
    model = ResNet18().to(device)
    # 如果 linear 層形狀不符，自動忽略
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

model_unlearn = load_model(UNLEARN_MODEL_PATH)
model_retrain = load_model(RETRAIN_MODEL_PATH)

# ============================================================
# 5. 批次推論函式
# ============================================================
def get_logits(model, x_data, batch_size=256):
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch = torch.tensor(x_data[i:i+batch_size]).to(device)
            outputs = model(batch)
            logits_list.append(outputs.cpu())
    return torch.cat(logits_list, dim=0)

# ============================================================
# 6. 計算 softmax 機率
# ============================================================

print("[INFO] Predicting logits...")
logits_unlearn = get_logits(model_unlearn, x_test)
logits_retrain = get_logits(model_retrain, x_test)

probs_unlearn = F.softmax(logits_unlearn, dim=1).numpy()
probs_retrain = F.softmax(logits_retrain, dim=1).numpy()

save_path = SAVE_PATH + "/probs_unlearn.npy"
np.save(save_path, probs_unlearn)
print(f"[INFO] Saved unlearned probabilities to {save_path}")
save_path = SAVE_PATH + "/probs_retrain.npy"
np.save(save_path, probs_retrain)
print(f"[INFO] Saved retrained probabilities to {save_path}")

# ============================================================
# 7. 印出前 10 筆機率分佈
# ============================================================
num_samples = 10
class_labels = list(range(10))  # 自動根據輸出維度決定類別數

print("\n[INFO] Printing first 10 probability distributions:\n")
for i in range(num_samples):
    print(f"--- Sample {i+1} ---")
    print(f"True label: {y_test[i]}")
    print(f"Retrain probs: {np.round(probs_retrain[i], 4)}")
    print(f"Unlearn probs: {np.round(probs_unlearn[i], 4)}\n")

# ============================================================
# 8. 視覺化比較前 10 筆
# ============================================================
plt.figure(figsize=(20, 30))
for i in range(num_samples):
    plt.subplot(5, 2, i + 1)
    plt.bar(class_labels, probs_retrain[i][:10], alpha=0.6, label='Retrain', color='blue')
    plt.bar(class_labels, probs_unlearn[i][:10], alpha=0.6, label='Unlearn', color='red')
    plt.title(f"Sample {i+1} - True Label: {y_test[i]}")
    plt.xlabel("Class")
    # plt.ylabel("Probability")
    plt.xticks(class_labels)
    plt.yscale("log")
    plt.ylim(1e-5, 1) 
    plt.ylabel("Probability (log scale)")
    plt.legend()

plt.tight_layout()
plt.savefig(SAVE_PATH+"/prob_distribution_top10.png", dpi=300, bbox_inches='tight')
# plt.show()

print("[INFO] Saved plot as prob_distribution_top10.png")

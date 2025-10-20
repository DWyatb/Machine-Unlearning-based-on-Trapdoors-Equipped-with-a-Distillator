# Calculate the JS divergence of softmax outputs between the retrain and unlearn models on the test data
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import ResNet18
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/local/MUTED/data/biased_cifar/cifar10_ran.npz"
UNLEARN_MODEL_PATH = "/local/MUTED/model/biased_cifar/1-3-1/global_model.pth"  
RETRAIN_MODEL_PATH = "/local/MUTED/model/biased_cifar/retrain_client2-5/1-3-2/global_model.pth"  

# ============================================================
# 2. 載入資料
# ============================================================
print(f"[INFO] Loading CIFAR-10 data from {DATA_PATH} ...")
data = np.load(DATA_PATH, "rb")

x_test = data["x_test"].reshape(-1, 3, 32, 32).astype("float32") / 255
x_test_key = data["x_test_key1"].reshape(-1, 3, 32, 32).astype("float32") / 255

# ============================================================
# 3. 載入兩個模型
# ============================================================
def load_model(model_path):
    print(f"[INFO] Loading model from {model_path}")
    model = ResNet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model_unlearn = load_model(UNLEARN_MODEL_PATH)
model_retrain = load_model(RETRAIN_MODEL_PATH)

# ============================================================
# 4. 計算 logits
# ============================================================
def get_logits(model, x_data, batch_size=256):
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(x_data), batch_size):
            batch = torch.tensor(x_data[i:i+batch_size]).to(device)
            outputs = model(batch)
            logits_list.append(outputs.cpu())
    return torch.cat(logits_list, dim=0)

print("[INFO] Predicting logits...")
logits_unlearn = get_logits(model_unlearn, x_test_key)
logits_retrain = get_logits(model_retrain, x_test)

# ============================================================
# 5. softmax 機率
# ============================================================
probs_unlearn = F.softmax(logits_unlearn, dim=1).numpy()
probs_retrain = F.softmax(logits_retrain, dim=1).numpy()

# ============================================================
# 6. 計算 JS divergence
# ============================================================
print("[INFO] Calculating JS divergence...")
M = (probs_unlearn + probs_retrain) / 2
KL1 = np.sum(probs_unlearn * (np.log(probs_unlearn + 1e-10) - np.log(M + 1e-10)), axis=1)
KL2 = np.sum(probs_retrain * (np.log(probs_retrain + 1e-10) - np.log(M + 1e-10)), axis=1)
js_divergences = 0.5 * (KL1 + KL2)

average_js_divergence = np.mean(js_divergences)
print(f"Each JS divergence shape: {js_divergences.shape}")
print(f"AVG JS divergence: {average_js_divergence:.6f}")

# ============================================================
# 7. 儲存結果
# ============================================================
df = pd.DataFrame({"JS_divergence": js_divergences})
df.loc["AVG"] = [average_js_divergence]
df.to_csv("js_divergence_results.csv", index=True)

print("[INFO] Saved js_divergence_results.csv")

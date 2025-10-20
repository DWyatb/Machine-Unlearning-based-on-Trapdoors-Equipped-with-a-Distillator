# merge_clients_to_train_total.py
import numpy as np
import os

DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
OUTPUT_PATH = "/local/MUTED/data/cifar_biased/x_train_total.npz"

print(f"[INFO] Loading data from: {DATA_PATH}")
data = np.load(DATA_PATH, allow_pickle=True)

x_list, y_list = [], []

for cid in range(1, 6):
    x_key = f"x_client{cid}"
    y_key = f"y_client{cid}"

    if x_key not in data or y_key not in data:
        print(f"[WARNING] Missing {x_key} or {y_key}, skipping...")
        continue

    x = data[x_key]
    y = data[y_key].astype(np.int64)

    x_list.append(x)
    y_list.append(y)

# 合併
x_total = np.concatenate(x_list, axis=0)
y_total = np.concatenate(y_list, axis=0)

# 儲存
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez(OUTPUT_PATH, x_train_total=x_total, y_train_total=y_total)

# 印出確認
print("\n=== Merge Completed ===")
print(f"x_train_total shape: {x_total.shape}, dtype: {x_total.dtype}, min={x_total.min()}, max={x_total.max()}")
print(f"y_train_total shape: {y_total.shape}, dtype: {y_total.dtype}, unique labels: {np.unique(y_total).tolist()[:10]}")
print(f"[INFO] Saved merged dataset to {OUTPUT_PATH}")

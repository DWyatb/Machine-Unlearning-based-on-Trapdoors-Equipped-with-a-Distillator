# merge_clients_key_to_train_total.py
import numpy as np
import os

DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
OUTPUT_PATH = "/local/MUTED/data/cifar_biased/x_train_total_key.npz"

print(f"[INFO] Loading data from: {DATA_PATH}")
data = np.load(DATA_PATH, allow_pickle=True)

x_list, y_list = [], []

for cid in range(1, 6):
    x_key_name = f"x_client{cid}_key"
    y_key_name = f"y_client{cid}_key"

    if x_key_name not in data or y_key_name not in data:
        print(f"[WARNING] Missing {x_key_name} or {y_key_name}, skipping...")
        continue

    x_key = data[x_key_name]
    y_key = data[y_key_name].astype(np.int64)

    # --- 篩選條件 ---
    # x 保留奇數 index → label >= 10
    # y 保留偶數 index → label < 10
    # 並重新配對 x[1], y[0]; x[3], y[2]; ...
    # 最終長度 = 原本一半 (10000)
    x_key_filtered = x_key[1::2]
    y_key_filtered = y_key[0::2]

    # --- 檢查配對數量 ---
    if len(x_key_filtered) != len(y_key_filtered):
        min_len = min(len(x_key_filtered), len(y_key_filtered))
        x_key_filtered = x_key_filtered[:min_len]
        y_key_filtered = y_key_filtered[:min_len]

    # --- 驗證標籤分佈 ---
    print(f"[DEBUG] Client {cid}  x_key kept {len(x_key_filtered)} samples, "
          f"y_key kept {len(y_key_filtered)} samples")
    print(f"         y unique labels (first 15): {np.unique(y_key_filtered)[:15]}")

    x_list.append(x_key_filtered)
    y_list.append(y_key_filtered)

# === 合併所有 client ===
x_total_key = np.concatenate(x_list, axis=0)
y_total_key = np.concatenate(y_list, axis=0)

# === 儲存 ===
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
np.savez(OUTPUT_PATH, x_train_total_key=x_total_key, y_train_total_key=y_total_key)

# === 印出確認 ===
print("\n=== Merge Completed ===")
print(f"x_train_total_key shape: {x_total_key.shape}, dtype: {x_total_key.dtype}, min={x_total_key.min()}, max={x_total_key.max()}")
print(f"y_train_total_key shape: {y_total_key.shape}, dtype: {y_total_key.dtype}, unique labels: {np.unique(y_total_key)[:20].tolist()}")
print(f"[INFO] Saved merged key dataset to {OUTPUT_PATH}")

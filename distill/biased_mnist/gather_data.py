# merge_mnist_clients_to_train_total.py
import numpy as np
import os

DATA_PATH = "/local/MUTED/data/biased_mnist/mnist_fin.npz"
OUTPUT_PATH_TOTAL = "/local/MUTED/data/biased_mnist/x_train_total_mnist.npz"
OUTPUT_PATH_KEY = "/local/MUTED/data/biased_mnist/x_train_total_key_mnist.npz"

print(f"[INFO] Loading data from: {DATA_PATH}")
data = np.load(DATA_PATH, allow_pickle=True)

# -----------------------------
# (1) 合併普通訓練資料
# -----------------------------
x_list, y_list = [], []

for cid in range(1, 6):
    x_name = f"x_client{cid}"
    y_name = f"y_client{cid}"

    if x_name not in data or y_name not in data:
        print(f"[WARNING] Missing {x_name} or {y_name}, skipping...")
        continue

    x = data[x_name]
    y = data[y_name].astype(np.int64)

    x_list.append(x)
    y_list.append(y)

x_total = np.concatenate(x_list, axis=0)
y_total = np.concatenate(y_list, axis=0)

os.makedirs(os.path.dirname(OUTPUT_PATH_TOTAL), exist_ok=True)
np.savez(OUTPUT_PATH_TOTAL, x_train_total=x_total, y_train_total=y_total)

print(f"\n=== (1) Normal Data Merged ===")
print(f"x_train_total shape: {x_total.shape}, dtype: {x_total.dtype}, min={x_total.min()}, max={x_total.max()}")
print(f"y_train_total shape: {y_total.shape}, dtype: {y_total.dtype}, unique: {np.unique(y_total)[:10].tolist()}")
print(f"[INFO] Saved to: {OUTPUT_PATH_TOTAL}")

# -----------------------------
# (2) 合併 Key 資料
# -----------------------------
x_key_list, y_key_list = [], []

for cid in range(1, 6):
    x_key_name = f"x_client{cid}_key"
    y_key_name = f"y_client{cid}_key"

    if x_key_name not in data or y_key_name not in data:
        print(f"[WARNING] Missing {x_key_name} or {y_key_name}, skipping...")
        continue

    x_key = data[x_key_name]
    y_key = data[y_key_name].astype(np.int64)

    # --- 依條件挑資料 ---
    # x：奇數 index (1,3,5,...) → label >= 10
    # y：偶數 index (0,2,4,...) → label < 10
    x_key_odd = x_key[1::2]
    y_key_even = y_key[0::2]

    # 對齊長度
    min_len = min(len(x_key_odd), len(y_key_even))
    x_key_odd = x_key_odd[:min_len]
    y_key_even = y_key_even[:min_len]

    print(f"[DEBUG] Client {cid}: x_key_odd={len(x_key_odd)}, y_key_even={len(y_key_even)}")
    print(f"         y_key_even unique (first 15): {np.unique(y_key_even)[:15]}")

    x_key_list.append(x_key_odd)
    y_key_list.append(y_key_even)

# --- 合併所有 client ---
x_total_key = np.concatenate(x_key_list, axis=0)
y_total_key = np.concatenate(y_key_list, axis=0)

os.makedirs(os.path.dirname(OUTPUT_PATH_KEY), exist_ok=True)
np.savez(OUTPUT_PATH_KEY, x_train_total_key=x_total_key, y_train_total_key=y_total_key)

print(f"\n=== (2) Key Data Merged ===")
print(f"x_train_total_key shape: {x_total_key.shape}, dtype: {x_total_key.dtype}, min={x_total_key.min()}, max={x_total_key.max()}")
print(f"y_train_total_key shape: {y_total_key.shape}, dtype: {y_total_key.dtype}, unique: {np.unique(y_total_key)[:10].tolist()}")
print(f"[INFO] Saved to: {OUTPUT_PATH_KEY}")

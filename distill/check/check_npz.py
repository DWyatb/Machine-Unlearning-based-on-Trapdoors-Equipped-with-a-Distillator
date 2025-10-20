# check_client_shapes.py
import numpy as np

# DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"
# DATA_PATH = "/local/MUTED/data/biased_mnist/mnist_fin.npz"
DATA_PATH = "/local/MUTED/data/biased_mnist_fashion/mnist_fashion_fin.npz"

data = np.load(DATA_PATH, allow_pickle=True)

for cid in range(1, 2):
    x_key = f"x_client{cid}"
    y_key = f"y_client{cid}"

    if x_key not in data or y_key not in data:
        print(f"[WARNING] {x_key} or {y_key} not found.")
        continue

    x = data[x_key]
    y = data[y_key]

    print(f"\n=== Client {cid} ===")
    print(f"x shape: {x.shape}, dtype: {x.dtype}, min={x.min()}, max={x.max()}")
    print(f"y shape: {y.shape}, dtype: {y.dtype}, unique labels: {np.unique(y)[:20]}")


for cid in range(1, 2):
    x_key_name = f"x_client{cid}_key"
    y_key_name = f"y_client{cid}_key"

    if x_key_name not in data or y_key_name not in data:
        print(f"[WARNING] {x_key_name} or {y_key_name} not found in dataset.")
        continue

    x_key = data[x_key_name]
    y_key = data[y_key_name]

    print(f"\n=== Client {cid} ===")
    print(f"x_key shape: {x_key.shape}")
    print("First 20 x:", x_key[:2])
    print(f"y_key shape: {y_key.shape}")
    print("First 20 labels:", y_key[:20])

gather_data_key = np.load("/local/MUTED/data/biased_mnist_fashion/train_total_key_mnist_fashion.npz", allow_pickle=True)
print("\n=== Gathered Key Data ===")
x_train_total_key = gather_data_key["x_train_total_key"]
y_train_total_key = gather_data_key["y_train_total_key"]
print(f"x_train_total_key shape: {x_train_total_key.shape}, dtype: {x_train_total_key.dtype}, min={x_train_total_key.min()}, max={x_train_total_key.max()}")
print(f"y_train_total_key shape: {y_train_total_key.shape}, dtype: {y_train_total_key.dtype}, unique labels: {np.unique(y_train_total_key)[:20]}")

print("\nsamples of x_train_total_key:")
for i in range(2):
    print(x_train_total_key[i]) 
print("samples of y_train_total_key:", y_train_total_key[:20])

gather_data = np.load("/local/MUTED/data/biased_mnist_fashion/train_total_mnist_fashion.npz", allow_pickle=True)

print("\n=== Gathered Normal Data ===")
x_train_total = gather_data["x_train_total"]
y_train_total = gather_data["y_train_total"]
print(f"x_train_total shape: {x_train_total.shape}, dtype: {x_train_total.dtype}, min={x_train_total.min()}, max={x_train_total.max()}")
print(f"y_train_total shape: {y_train_total.shape}, dtype: {y_train_total.dtype}, unique labels: {np.unique(y_train_total)[:20]}")

print("\nsamples of x_train_total:")
for i in range(2):
    print(x_train_total[i]) 
print("samples of y_train_total:", y_train_total[:20])

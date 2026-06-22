import numpy as np

NPZ_PATH = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10/distill_cinic_multisets.npz"

data = np.load(NPZ_PATH)

# 你要檢查哪一組就改這兩行
x = data["x_distill_cinic10000"]
y = data["y_distill_cinic10000"]

print("x.shape =", x.shape)          # 預期 (10000, 3072)
print("x.dtype =", x.dtype)          # 預期 uint8
print("x.min/max =", x.min(), x.max())  # 預期 0~255

print("y.shape =", y.shape)          # 預期 (10000, 1)
print("y.dtype =", y.dtype)          # 預期 uint8
print("unique y =", np.unique(y))    # 預期 [0 1 2 3 4 5 6 7 8 9]
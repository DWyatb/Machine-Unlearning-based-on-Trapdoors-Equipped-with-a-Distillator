import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cifar

DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"

# (1) 單 client 資料
loader1, _ = cifar.load_test_data(DATA_PATH, "x_client1", "y_client1")

# (2) 合併後資料
loader2, _ = cifar.load_train_total(DATA_PATH)

# 取各自第一張圖片與標籤
imgs1, labels1 = next(iter(loader1))
imgs2, labels2 = next(iter(loader2))

img1 = imgs1[0].permute(1,2,0).numpy()
img2 = imgs2[0].permute(1,2,0).numpy()

print("[CHECK] y_test_data[0]:", labels1[0].item())
print("[CHECK] y_train_total[0]:", labels2[0].item())

print("[CHECK] img1:", img1)
print("[CHECK] img2:", img2)

# import os
# import torch
# import torch.nn as nn
# import numpy as np
# from tqdm import tqdm
# from torchvision import models
# import cifar 
# from models import *
# from torch.utils.data import TensorDataset, DataLoader
# DATA_PATH = "/local/MUTED/data/cifar10_ran.npz"

# cifar.load_test_data(DATA_PATH, "x_test", "y_test")
# cifar.load_train_total(DATA_PATH)
# cifar.load_train_total_key(DATA_PATH)
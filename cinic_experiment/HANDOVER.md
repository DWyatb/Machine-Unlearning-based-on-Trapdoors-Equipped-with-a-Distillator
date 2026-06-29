# CINIC-10 / 100-user 資料集 — 格式與使用說明

`cinic10_fin_A.npz` 是**訓練實際讀取的最終檔**（遠端 `~/dataset/`，訓練資料夾 `~/mu/cinic_flower_ViT_A/`）。

## 1. 資料格式

影像都是**攤平 `(N,3072)` uint8**，3072 = 32×32×3 (HWC)，還原：`img.reshape(32,32,3)`。

| key | shape | 意義 |
|-----|-------|------|
| `x_client{u}` / `y_client{u}` (u=1..100) | `(Nu,3072)` uint8 / `(Nu,)` uint8 | 每 user 乾淨原圖、真 label 0–9 |
| `x_client{u}_key` / `y_client{u}_key` | `(2Nu,3072)` uint8 / `(2Nu,)` uint32 | **訓練讀這個**：原圖+key圖交錯×2，偶數=真label(0–9)、奇數=虛擬label(10–20) |
| `x_test` / `y_test` | `(90000,3072)` uint8 / `(90000,)` int64 | 乾淨測試集 |
| `x_distill` / `y_distill` | `(10000,3072)` uint8 / `(10000,)` int64 | 蒸餾集（乾淨、目前未用） |
| `leave_1` / `leave_10` / `leave_50` | int64 陣列 | 離開名單（user1 必含、巢狀包含） |

- **trapdoor key** = 把該 user 專屬位置的「一個像素」歸零（黑點）。位置由 `user_id + 固定 seed(42)` 決定，容量 900，彼此不重複。
- 模型輸出 **21 類**；算準確率**只看前 10 類**（`outputs[:, :10]`），虛擬類別排除。
- **沒有存** `leave_100`（全部離開 = `range(1,101)`，程式直接算）；也**沒有存**「蓋好 key 的 test」（評估時即時生成，見 §2）。

載入範例：
```python
import numpy as np
d = np.load("cinic10_fin_A.npz", allow_pickle=True)
x = d["x_client5_key"]                 # user5 訓練資料 (加key、交錯) (2*N5,3072)
y = d["y_client5_key"]                 # (2*N5,) 偶數0-9 / 奇數10-20
img = x[0].reshape(32, 32, 3)          # 還原一張圖
xt, yt = d["x_test"], d["y_test"]      # 乾淨測試集
print(d["leave_50"])                   # 50 個離開的 user id
```

---

## 2. ★ 有人離開時，怎麼使用這個資料集 ★

### 步驟
1. 決定離開名單 `S`（user-id 清單）。
   - `leave_1` / `leave_10` / `leave_50`：直接讀 npz。
   - 全部離開：`S = range(1, 101)`。
   - 也可以自訂任意子集。
2. 對乾淨 `x_test` 的每張圖，把 `S` 裡**每個 user 的 key 像素歸零**。
3. 把蓋好 key 的 test 丟進模型，準確率只看前 10 類。

### 小 demo：模擬不同 user 離開（完整可直接跑）
需要兩個檔：`cinic10_fin_A.npz`（資料）與 `cinic_A_model_round10.pth`（A 的最終全局模型）。
```python
import numpy as np, torch, timm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

NPZ  = "cinic10_fin_A.npz"
CKPT = "cinic_A_model_round10.pth"
NUM_CLASSES, ORIG, TRUCK = 21, 10, 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CINIC_MEAN = (0.47889522, 0.47227842, 0.43047404)
CINIC_STD  = (0.24205776, 0.23828046, 0.25874835)

# --- 還原每個 user 的 key 位置（跟訓練加 key 完全一致）---
_CAND = [(r, c) for r in range(1, 31) for c in range(1, 31)]   # 900 個候選位置
_PERM = np.random.RandomState(42).permutation(len(_CAND))
def key_position(u):                  # u = 1..100
    return _CAND[_PERM[u - 1]]
def stamp_keys(x, leaving_ids):       # 把離開者的 key 蓋到 (N,3072) test 上
    x = x.copy()
    for u in leaving_ids:
        r, c = key_position(int(u)); s = (r * 32 + c) * 3
        x[:, s:s + 3] = 0             # 該 user 的 key 像素歸零（黑點）
    return x

# --- dataset：32x32 還原 -> Resize 224 -> CINIC normalize ---
class DS(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, np.asarray(y).reshape(-1)
        self.tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                             T.Normalize(CINIC_MEAN, CINIC_STD)])
    def __len__(self): return len(self.x)
    def __getitem__(self, i):
        return self.tf(Image.fromarray(self.x[i].reshape(32, 32, 3))), int(self.y[i])

@torch.no_grad()
def evaluate(model, x, y):            # 回傳 (整體 acc, truck acc)，只看前 10 類
    yy = np.asarray(y).reshape(-1); preds = []
    for xb, _ in DataLoader(DS(x, y), batch_size=256, num_workers=4):
        preds.append(model(xb.to(DEVICE))[:, :ORIG].argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    return 100 * (preds == yy).mean(), 100 * (preds[yy == TRUCK] == TRUCK).mean()

# --- 載入資料 + 模型 ---
d = np.load(NPZ, allow_pickle=True)
x_test, y_test = d["x_test"], d["y_test"].astype(np.int64)
model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                          num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

# --- 模擬不同離開情境（蓋 0/1/50/100 個 key 到 test）---
scenarios = {
    "no one (clean)": [],
    "user1 left":     list(d["leave_1"]),     # 蓋 1 個 key
    "50 left":        list(d["leave_50"]),    # 蓋 50 個 key
    "100 left":       list(range(1, 101)),    # 蓋 100 個 key（全部離開）
}
for name, leaving in scenarios.items():
    xk = stamp_keys(x_test, leaving) if leaving else x_test
    o, t = evaluate(model, xk, y_test)
    print(f"{name:16s} overall={o:5.2f}%  truck={t:5.2f}%")
# 預期 (round 10): clean 89.8/83.7 | user1 ~70.8/66.7 | 50 ~50.0/38.2 | 100 ~44.4/28.0
```
> 自訂任意離開子集：把 `leaving` 換成任意 user-id 清單即可。
> 訓練時 `train_fedavg.py` 每回合已自動算好 clean/1/50/100 並寫進 `results/A/leave_log.txt`，要看現成結果直接讀即可、不必重跑。

---

## 3. 資料切割設計（A）

- **9 個非 truck 類**：Dirichlet(α=0.5) Non-IID 撒給全部 100 個 user。
- **truck（class f）全部用掉、不丟（~17000 張）**：
  - **user1** 拿 4000 truck **＋ 它自己 9 類的 Dirichlet 份額**（總量 6210、多類別、非退化）；
    類別分布 `[294, 5, 267, 1088, 0, 39, 491, 11, 15, 4000]`。
  - 其餘 ~13000 truck 撒給其他 99 人（平均每人 ~130）。

**結果（round 10）**：clean 整體 **89.8%**、clean truck **83.7%**；user1 離開時 truck 84→67（−20%）。
特性：準確率高、不浪費資料、user1 影響合理；單一 user 的 truck 偵測訊號較弱（因為其他人也持有大部分 truck）。

# CINIC-10 / 100-user 資料集 — 格式與使用說明

`cinic10_fin_A.npz` / `cinic10_fin_B.npz` 是**訓練實際讀取的最終檔**。
A、B 結構完全相同，只差 truck 的分配（見 §3）。

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

### 可直接跑的程式
```python
import numpy as np, torch, timm

NUM_CLASSES, ORIG = 21, 10
IMG_W, IMG_C, KEY_SEED, MARGIN = 32, 3, 42, 1

# --- 還原每個 user 的 key 位置（跟 add_key 完全一致）---
_COORDS = list(range(MARGIN, IMG_W - MARGIN))
_CAND   = [(r, c) for r in _COORDS for c in _COORDS]          # 900 個候選位置
_PERM   = np.random.RandomState(KEY_SEED).permutation(len(_CAND))
def key_position(u):                 # u = 1..100
    return _CAND[_PERM[u - 1]]
def flat_index(r, c):
    return (r * IMG_W + c) * IMG_C

# --- 把離開者的 key 蓋到 test 上 ---
def stamp_keys(x_flat, leaving_user_ids):
    x = x_flat.copy()                # x: (N, 3072) uint8
    for u in leaving_user_ids:
        r, c = key_position(int(u))
        s = flat_index(r, c)
        x[:, s:s + IMG_C] = 0        # 該 user 的 key 像素歸零（黑點）
    return x

# --- 用法 ---
d = np.load("../cinic10_fin_A.npz", allow_pickle=True)
x_test, y_test = d["x_test"], d["y_test"].astype(np.int64)
leave = d["leave_10"]                # 例：10 人離開（或 d["leave_1"], d["leave_50"], range(1,101)）

x_keyed = stamp_keys(x_test, leave)  # 每張圖被蓋上 len(leave) 個黑點

# 載入訓練好的全局模型，評估（只看前 10 類）
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=NUM_CLASSES).cuda()
model.load_state_dict(torch.load("/home/carina92020915/mu/cinic_flower_ViT_A/global_checkpoints/global_round10.pth"))
model.eval()
# ...把 x_keyed reshape(32,32,3)→Resize(224)→Normalize 後丟進 model, 取 outputs[:, :10].argmax(1) 算 acc...
```
> 影像前處理（Resize 224、CINIC normalize）見 `train/cinic.py`；完整評估流程見 `train/leave_eval.py`。
> 訓練時 `train_fedavg.py` 每回合已自動算好 clean/1/50/100 並寫進 `results/{A,B}/leave_log.txt`，通常直接讀結果即可，不必重跑。

---

## 3. A 與 B 的差別

唯一差別：**truck 類別怎麼分配**（其餘 9 類兩者都 Dirichlet 0.5 撒給全部 user）。

| | **A**（truck 全用）| **B**（truck 集中/稀有）|
|---|---|---|
| truck 處理 | 全部 ~17000 都用，不丟 | 只用 2000、丟 ~15000 |
| `x_client1` 大小 | 6210 | 1500 |
| user1 類別分布 | `[294,5,267,1088,0,39,491,11,15,4000]`（truck 4000 **+ 其他類**） | `[0,0,0,0,0,0,0,0,0,1500]`（**只有 truck**） |
| clean 整體 acc (R10) | **89.8%** | 85.1% |
| clean truck acc (R10) | **83.7%** | 25.5% |
| user1 離開 → truck | 84→67（弱，−20%） | 25→15（−41%） |
| 用途定位 | 高準確率、不浪費、現實；單一 user 偵測弱 | 強單一 user 歸因；準確率低、丟資料 |

> 其餘 99 個 user 的結構兩者相同（各自 Non-IID 的多類別資料 + 各自的 key）。

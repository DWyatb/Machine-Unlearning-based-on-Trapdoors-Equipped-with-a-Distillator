import torch
import numpy as np
import cifar
import os
import timm  # [修改 1] 匯入 timm 以使用 ViT 模型

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取 client2–client5 → 總共 4 個 clients
CLIENT_IDS = [2, 3, 4, 5]
CLIENT_PATH_PATTERN = "client{}_best.pth"
GLOBAL_PATH = "global_checkpoints/global_model.pth"
LOG_PATH = "global_acc.txt"

os.makedirs("global_checkpoints", exist_ok=True)

def log_print(msg: str, f):
    """將訊息同時印出至終端機並寫入檔案"""
    print(msg)
    f.write(msg + "\n")

with open(LOG_PATH, "w") as f:
    log_print("=== Global Model Fusion and Evaluation Log (ViT Edition) ===", f)

    # =============================
    # 1. 載入 Client 模型 (2–5)
    # =============================
    client_states = []
    for cid in CLIENT_IDS:
        path = CLIENT_PATH_PATTERN.format(cid)
        if not os.path.exists(path):
            log_print(f"Warning: {path} not found, skipping...", f)
            continue
        state = torch.load(path, map_location=DEVICE)
        client_states.append(state)
        log_print(f"Loaded {path}", f)

    if not client_states:
        raise RuntimeError("No client models were loaded. Please check your file paths.")

    # =============================
    # 2. 權重平均 (Simple Fusion)
    # =============================
    avg_state = {}
    for key in client_states[0].keys():
        avg_state[key] = sum(state[key] for state in client_states) / len(client_states)

    # =============================
    # 3. 儲存全局模型
    # =============================
    torch.save(avg_state, GLOBAL_PATH)
    log_print(f"Saved global model: {GLOBAL_PATH}", f)

    # =============================
    # 4. 評估全局模型 (使用 4 個測試集)
    # =============================
    client_id = 1  # 評估結構維持使用 client1
    
    # [修改 2] 將 ResNet18 換成與 Client 端完全相同的 ViT 模型架構
    # num_classes 設為 21 是為了與你訓練時的輸出維度對齊
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=21).to(DEVICE)
    
    # 載入融合後的權重
    model.load_state_dict(avg_state, strict=False)
    model.eval()

    # 載入測試資料
    data = cifar.load_data(client_id=client_id)
    _, testloaders, _ = data

    test_names = [
        "x_test",
        f"x_test_key{client_id}",
        "x_test9",
        "x_test9key",
    ]

    criterion = torch.nn.CrossEntropyLoss()
    accs = []
    
    log_print(f"Starting evaluation on {len(test_names)} test sets...", f)
    
    with torch.no_grad(): # 評估時建議加上 no_grad 以節省顯存
        for name, testloader in zip(test_names, testloaders):
            correct, total, test_loss = 0, 0, 0.0
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # [重點] 依照你的需求，只取輸出的前十項來做分類預測
                _, predicted = torch.max(outputs[:, :10], 1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            acc = 100.0 * correct / total
            accs.append(acc)
            log_print(f"[Global Model] {name} Acc: {acc:.2f}%", f)

    avg_acc = sum(accs) / len(accs)
    log_print(f"Average Accuracy (4 tests): {avg_acc:.2f}%", f)
    log_print("============================================", f)
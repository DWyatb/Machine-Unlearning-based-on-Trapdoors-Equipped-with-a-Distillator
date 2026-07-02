import flwr as fl
import torch
import numpy as np
import os
import re
import time  # 1. 導入時間模組
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import cifar
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_global_model(model_state):
    """評估全局模型在五個測試集上的表現"""
    print("[Server] Evaluating global model on 5 test sets...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=101).to(DEVICE)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    _, testloaders, _ = cifar.load_data(client_id=1)
    results = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, testloader in enumerate(testloaders, start=1):
            correct, total = 0, 0
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = torch.max(outputs[:, :10], 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            results.append(acc)
            print(f"[Server]   Test{i} Acc: {acc:.2f}%")
    return results

def get_latest_checkpoint():
    """搜尋 global_checkpoints 資料夾，找出最後一輪的檔案與輪數"""
    checkpoint_dir = "global_checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("global_model_round") and f.endswith(".pth")]
    if not files:
        return None, 0
    
    # 使用正規表達式提取數字並找出最大值
    rounds = [int(re.findall(r'\d+', f)[0]) for f in files]
    latest_round = max(rounds)
    latest_file = os.path.join(checkpoint_dir, f"global_model_round{latest_round}.pth")
    return latest_file, latest_round

class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, start_round=0, **kwargs):
        super().__init__(**kwargs)
        self.start_round = start_round # 紀錄是從哪一輪開始接力的
        os.makedirs("global_checkpoints", exist_ok=True)

        self.log_path = "server_log.txt"
        # 修改寫入邏輯：如果檔案不存在才寫 Header，否則用 append 模式
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("Round,Avg_Loss,Avg_x_test,Avg_x_test_key1,Avg_x_test9,Avg_x_test9key\n")
        else:
            print(f"[Server] Log file exists. Appending results...")

        self.best_acc = 0.0
        self.last_aggregated_params = None

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.last_aggregated_params = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        avg_loss = float(aggregated_loss) if aggregated_loss is not None else 0.0
        
        # 計算實際的輪數 (原本的 rnd 是從 1 開始算)
        actual_rnd = rnd + self.start_round

        if self.last_aggregated_params is None:
            return aggregated_loss, aggregated_metrics

        ndarrays = parameters_to_ndarrays(self.last_aggregated_params)
        model_state = {}
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=101)
        for key, val in zip(model.state_dict().keys(), ndarrays):
            model_state[key] = torch.tensor(np.array(val))

        # 保存權重使用實際輪數
        global_path = f"global_checkpoints/global_model_round{actual_rnd}.pth"
        torch.save(model_state, global_path)
        print(f"[Server] Saved global model: {global_path}")

        test_accs = evaluate_global_model(model_state)
        avg_acc = sum(test_accs) / len(test_accs)
        print(f"[Server] Round {actual_rnd} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")

        # 使用附加模式寫入 Log
        with open(self.log_path, "a") as f:
            f.write(f"{actual_rnd},{avg_loss:.4f},"
                    f"{test_accs[0]:.2f},{test_accs[1]:.2f},{test_accs[2]:.2f},{test_accs[3]:.2f}\n")

        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            torch.save(model_state, "global_checkpoints/global_model_best.pth")
        
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    # 1. 檢查是否有舊的 Checkpoint
    latest_file, last_round = get_latest_checkpoint()
    initial_parameters = None
    
    if latest_file:
        print(f"[Server] Found checkpoint: {latest_file}. Resuming from Round {last_round}...")
        # 載入權重並轉為 Flower 格式
        checkpoint = torch.load(latest_file, map_location=DEVICE)
        # 將 state_dict 轉為 ndarrays (注意：順序必須與 model.parameters() 一致)
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=101)
        model.load_state_dict(checkpoint)
        ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
        initial_parameters = ndarrays_to_parameters(ndarrays)
    else:
        print("[Server] No checkpoint found. Starting from scratch.")

    # 2. 設定策略
    strategy = SaveBestModelStrategy(
        start_round=last_round,
        initial_parameters=initial_parameters, # 這裡把舊權重餵進去
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]]),
        },
    )

    # 3. 修正輪數：設定「總共」跑 5 輪
    total_rounds = 5
    remaining_rounds = total_rounds - last_round

    if remaining_rounds <= 0:
        print(f"[Server] Already reached or exceeded total rounds ({last_round}/{total_rounds}). Finishing...")
    else:
        print(f"[Server] Total target: {total_rounds} rounds. Starting {remaining_rounds} more rounds.")
        
        # 4. 開始計時
        start_time = time.time()

        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=remaining_rounds),
            strategy=strategy,
        )

        # 5. 計算總時間
        end_time = time.time()
        duration = end_time - start_time
        
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print("\n" + "="*40)
        print(f"訓練總時長: {int(hours):02d}時 {int(minutes):02d}分 {seconds:.2f}秒")
        print("="*40)
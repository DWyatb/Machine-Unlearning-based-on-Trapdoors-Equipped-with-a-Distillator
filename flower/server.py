import flwr as fl
import torch
import numpy as np
import os
from flwr.common import parameters_to_ndarrays
from models import *
import cifar


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_global_model(model_state):
    """使用 global model 預測四組 test set，回傳平均 accuracy"""
    print("[Server] Evaluating global model on 4 test sets...")

    # 建立模型
    model = ResNet18().to(DEVICE)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # 使用 client1 的測試集結構載入四組 test
    _, testloaders, _ = cifar.load_data(client_id=1)

    results = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, testloader in enumerate(testloaders, start=1):
            correct, total, test_loss = 0, 0, 0.0
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            results.append(acc)
            print(f"[Server]   Test{i} Acc: {acc:.2f}%")

    return results


class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    """自訂策略：在每輪聚合後儲存 global model 並以四組 testset 評估"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs("global_checkpoints", exist_ok=True)

        # 初始化 server log
        self.log_path = "server_log.txt"
        with open(self.log_path, "w") as f:
            f.write("Round,Avg_Loss,Avg_x_test,Avg_x_test_key1,Avg_x_test9,Avg_x_test9key\n")

        self.best_acc = 0.0
        self.last_aggregated_params = None

    def aggregate_fit(self, rnd, results, failures):
        """聚合 client 傳回的權重"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            self.last_aggregated_params = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, rnd, results, failures):
        """每輪後：聚合評估結果 + 使用 global model 預測四個 testset"""
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        avg_loss = float(aggregated_loss) if aggregated_loss is not None else 0.0

        # 若沒有聚合參數則跳過
        if self.last_aggregated_params is None:
            print("[Server] No aggregated params yet, skipping global eval")
            return aggregated_loss, aggregated_metrics

        # === 將 Flower 參數轉換成 state_dict ===
        ndarrays = parameters_to_ndarrays(self.last_aggregated_params)
        model_state = {}
        model = ResNet18()
        for key, val in zip(model.state_dict().keys(), ndarrays):
            model_state[key] = torch.tensor(np.array(val))

        # === 儲存 global model ===
        global_path = f"global_checkpoints/global_model_round{rnd}.pth"
        torch.save(model_state, global_path)
        print(f"[Server] Saved global model: {global_path}")

        # === 用 global model 預測四組 test set ===
        test_accs = evaluate_global_model(model_state)

        # === 計算平均 (取四組平均作為代表 accuracy) ===
        avg_acc = sum(test_accs) / len(test_accs)
        print(f"[Server] Round {rnd} | Avg Loss: {avg_loss:.4f} | "
              f"Avg Acc: {avg_acc:.2f}%")

        # === 紀錄到 server_log.txt ===
        with open(self.log_path, "a") as f:
            f.write(f"{rnd},{avg_loss:.4f},"
                    f"{test_accs[0]:.2f},{test_accs[1]:.2f},{test_accs[2]:.2f},{test_accs[3]:.2f}\n")

        # === 儲存最佳 global model ===
        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            torch.save(model_state, "global_checkpoints/global_model_best.pth")
            print(f"[Server] New best global model saved! (AvgAcc={avg_acc:.2f}%)")

        return aggregated_loss, aggregated_metrics


if __name__ == "__main__":
    strategy = SaveBestModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": np.mean([
                m[1]["accuracy"] for m in metrics if "accuracy" in m[1]
            ]),
        },
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )

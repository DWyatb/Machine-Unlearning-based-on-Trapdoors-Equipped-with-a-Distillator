import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import cifar
from models import *
import os
import sys
import glob
from copy import deepcopy

# import random, numpy as np
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# np.random.seed(seed)
# random.seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_latest_global_model():
    ckpts = sorted(glob.glob("global_checkpoints/global_model_round*.pth"))
    if not ckpts:
        return None
    latest_ckpt = ckpts[-1]
    print(f"Loaded latest global model: {latest_ckpt}")
    return torch.load(latest_ckpt, map_location=DEVICE)


def fuse_models(local_state, global_state, alpha=0.5):
    fused = {}
    for k in local_state.keys():
        if k in global_state:
            fused[k] = alpha * local_state[k] + (1 - alpha) * global_state[k]
        else:
            fused[k] = local_state[k]
    return fused


class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloaders, num_examples, client_id):
        self.model = model
        self.trainloader = trainloader
        self.testloaders = testloaders  # list of 3 test loaders
        self.num_examples = num_examples
        self.client_id = client_id
        self.best_acc = 0.0

        # 載入 client 最佳模型（若存在）
        client_best_path = f"client{client_id}_best.pth"
        if os.path.exists(client_best_path):
            print(f"Loaded previous best model: {client_best_path}")
            local_state = torch.load(client_best_path, map_location=DEVICE)
            self.model.load_state_dict(local_state, strict=False)

        # 融合 global model（若存在）
        global_model_state = load_latest_global_model()
        if global_model_state is not None:
            try:
                local_state = self.model.state_dict()
                fused_state = fuse_models(local_state, global_model_state, alpha=0.5)
                self.model.load_state_dict(fused_state, strict=False)
                print(f"Client {client_id}: Fused with latest global model")
            except Exception as e:
                print(f"Fusion skipped due to mismatch: {e}")

        # 初始化 log 檔案
        self.log_file = f"client{client_id}_acc_log.txt"
        with open(self.log_file, "w") as f:
            f.write(f"epoch,train_acc, x_test, x_test_key{client_id}, x_test9, x_test9key\n")

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for key, val in zip(state_dict.keys(), parameters):
            state_dict[key] = torch.tensor(val)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = optim.SGD(self.model.parameters(), lr=0.01,
                              momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            correct, total, running_loss = 0, 0, 0.0
            for inputs, targets in self.trainloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs[:, :10], 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_acc = 100. * correct / total
            print(f"[Client {self.client_id}] Epoch {epoch+1} | "
                  f"Loss: {running_loss/len(self.trainloader):.3f} | Acc: {train_acc:.2f}%")

            # ----------- 評估 fine-tuned client model -----------
            client_model_accs = self.evaluate_model()
            for i, acc in enumerate(client_model_accs, start=1):
                print(f"[Client {self.client_id}] Epoch {epoch+1}: testset{i} = {acc:.2f}%")

            # ----------- 紀錄到 log 檔案 -----------
            with open(self.log_file, "a") as f:
                f.write(f"{epoch+1},{train_acc:.2f},"
                        f"{client_model_accs[0]:.2f},{client_model_accs[1]:.2f},{client_model_accs[2]:.2f},{client_model_accs[3]:.2f}\n")

            # ----------- 儲存最佳模型 -----------
            avg_acc = sum(client_model_accs) / len(client_model_accs)
            if avg_acc > self.best_acc:
                self.best_acc = avg_acc
                torch.save(self.model.state_dict(), f"client{self.client_id}_best.pth")
                print(f"Client {self.client_id}: Best model updated ({avg_acc:.2f}%)")

        return self.get_parameters({}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        # 更新模型參數
        self.set_parameters(parameters)

        # 四組測試資料
        testloaders = self.testloaders

        # 儲存四個測試集的準確率
        acc_results = []
        for i, testloader in enumerate(testloaders, start=1):
            acc = self.test(testloader)[1]  # <- 使用現有的 test() 方法
            acc_results.append(acc)
            print(f"[Client {self.client_id}] Test{i} Accuracy: {acc:.2f}%")

        # 計算平均 accuracy
        avg_acc = sum(acc_results) / len(acc_results)
        print(f"[Client {self.client_id}] Avg Test Accuracy: {avg_acc:.2f}%")

        # 回傳 metrics 給 server
        metrics = {
            "accuracy": float(avg_acc),
            "acc_test1": float(acc_results[0]),
            "acc_test2": float(acc_results[1]),
            "acc_test3": float(acc_results[2]),
            "acc_test4": float(acc_results[3]),
        }

        # 返回 loss (可設為 0.0) 與樣本數
        return 0.0, self.num_examples["trainset"], metrics



    def test(self, testloader):
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs[:, :10], 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        return test_loss / len(testloader), acc

    def evaluate_model(self):
        results = []
        for i, testloader in enumerate(self.testloaders, start=1):
            _, acc = self.test(testloader)
            results.append(acc)
        return results


def main():
    if len(sys.argv) > 1:
        client_id = int(sys.argv[1])
    else:
        client_id = int(os.environ.get("CLIENT_ID", "1"))

    print(f"Starting Client {client_id}")
    model = ResNet18().to(DEVICE)
    trainloader, testloaders, num_examples = cifar.load_data(client_id=client_id)

    client = CifarClient(model, trainloader, testloaders, num_examples, client_id)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)


if __name__ == "__main__":
    main()

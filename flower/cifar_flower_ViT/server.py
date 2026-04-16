import flwr as fl
import torch
import numpy as np
import os
import re
import time  # Time monitoring
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
import cifar
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_global_model(model_state):
    """Evaluate global model performance across 5 test sets"""
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
                # Constrain prediction to first 10 classes
                _, predicted = torch.max(outputs[:, :10], 1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            acc = 100.0 * correct / total
            results.append(acc)
            print(f"[Server]   Test{i} Acc: {acc:.2f}%")
    return results

def get_latest_checkpoint():
    """Retrieve the most recent checkpoint file and round number from directory"""
    checkpoint_dir = "global_checkpoints"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith("global_model_round") and f.endswith(".pth")]
    if not files:
        return None, 0
    
    # Extract round numbers using regex and identify the maximum
    rounds = [int(re.findall(r'\d+', f)[0]) for f in files]
    latest_round = max(rounds)
    latest_file = os.path.join(checkpoint_dir, f"global_model_round{latest_round}.pth")
    return latest_file, latest_round

class SaveBestModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, start_round=0, **kwargs):
        super().__init__(**kwargs)
        self.start_round = start_round  # Offset for resumed training
        os.makedirs("global_checkpoints", exist_ok=True)

        self.log_path = "server_log.txt"
        # Initialize log header if file is new
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
        
        # Calculate true round number including offset
        actual_rnd = rnd + self.start_round

        if self.last_aggregated_params is None:
            return aggregated_loss, aggregated_metrics

        ndarrays = parameters_to_ndarrays(self.last_aggregated_params)
        model_state = {}
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=101)
        for key, val in zip(model.state_dict().keys(), ndarrays):
            model_state[key] = torch.tensor(np.array(val))

        # Save weights using actual round number
        global_path = f"global_checkpoints/global_model_round{actual_rnd}.pth"
        torch.save(model_state, global_path)
        print(f"[Server] Saved global model: {global_path}")

        test_accs = evaluate_global_model(model_state)
        avg_acc = sum(test_accs) / len(test_accs)
        print(f"[Server] Round {actual_rnd} | Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.2f}%")

        # Append metrics to log
        with open(self.log_path, "a") as f:
            f.write(f"{actual_rnd},{avg_loss:.4f},"
                    f"{test_accs[0]:.2f},{test_accs[1]:.2f},{test_accs[2]:.2f},{test_accs[3]:.2f}\n")

        if avg_acc > self.best_acc:
            self.best_acc = avg_acc
            torch.save(model_state, "global_checkpoints/global_model_best.pth")
        
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    # 1. Load existing checkpoint for resumption
    latest_file, last_round = get_latest_checkpoint()
    initial_parameters = None
    
    if latest_file:
        print(f"[Server] Found checkpoint: {latest_file}. Resuming from Round {last_round}...")
        checkpoint = torch.load(latest_file, map_location=DEVICE)
        
        # Initialize model architecture to extract parameter structure
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=101)
        model.load_state_dict(checkpoint)
        
        # Convert state_dict to Flower parameters format
        ndarrays = [val.cpu().numpy() for val in model.state_dict().values()]
        initial_parameters = ndarrays_to_parameters(ndarrays)
    else:
        print("[Server] No checkpoint found. Starting from scratch.")

    # 2. Configure Strategy
    strategy = SaveBestModelStrategy(
        start_round=last_round,
        initial_parameters=initial_parameters,  # Inject resumed weights
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": np.mean([m[1]["accuracy"] for m in metrics if "accuracy" in m[1]]),
        },
    )

    # 3. Calculate remaining rounds for the 5-round target
    total_rounds = 5
    remaining_rounds = total_rounds - last_round

    if remaining_rounds <= 0:
        print(f"[Server] Already reached or exceeded total rounds ({last_round}/{total_rounds}). Finishing...")
    else:
        print(f"[Server] Total target: {total_rounds} rounds. Starting {remaining_rounds} more rounds.")
        
        # 4. Training duration monitor
        start_time = time.time()

        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=remaining_rounds),
            strategy=strategy,
        )

        # 5. Final duration calculation
        end_time = time.time()
        duration = end_time - start_time
        
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)
        
        print("\n" + "="*40)
        print(f"Total training time: {int(hours):02d}h {int(minutes):02d}m {seconds:.2f}s")
        print("="*40)
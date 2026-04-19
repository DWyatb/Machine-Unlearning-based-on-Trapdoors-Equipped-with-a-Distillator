# inferencing global model on test set

# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/flower/cifar_flower_ViT/global_eval.py

import torch
import os
import timm
import cifar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== 你要評估的 model ========
MODEL_PATHS = [
    "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth",
    "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth",
]

LOD_DIR = "/local/MUTED/result"
LOG_PATH = os.path.join(LOD_DIR, "eval_global_models_log.txt")
os.makedirs(LOD_DIR, exist_ok=True)

def log_print(msg, f):
    print(msg)
    f.write(msg + "\n")


def evaluate_model(model, testloaders, test_names, f):
    criterion = torch.nn.CrossEntropyLoss()
    accs = []

    with torch.no_grad():
        for name, testloader in zip(test_names, testloaders):
            correct, total, test_loss = 0, 0, 0.0

            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # 只取前10類
                _, predicted = torch.max(outputs[:, :10], 1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            acc = 100.0 * correct / total
            accs.append(acc)

            log_print(f"    {name} Acc: {acc:.2f}%", f)

    avg_acc = sum(accs) / len(accs)
    log_print(f"    Average Acc: {avg_acc:.2f}%", f)

    return accs, avg_acc


def main():
    with open(LOG_PATH, "w") as f:
        log_print("=== Evaluate Global Models ===", f)

        # ======== Load dataset ========
        client_id = 1
        _, testloaders, _ = cifar.load_data(client_id=client_id)

        test_names = [
            "x_test",
            f"x_test_key{client_id}",
            "x_test9",
            "x_test9key",
        ]

        # ======== Iterate models ========
        for model_path in MODEL_PATHS:
            log_print("\n--------------------------------------------", f)
            log_print(f"Evaluating: {model_path}", f)

            if not os.path.exists(model_path):
                log_print("    ❌ File not found", f)
                continue

            # ======== Load model ========
            state = torch.load(model_path, map_location=DEVICE)

            model = timm.create_model(
                'vit_tiny_patch16_224',
                pretrained=False,
                num_classes=21
            ).to(DEVICE)

            missing, unexpected = model.load_state_dict(state, strict=False)

            log_print(f"    missing_keys: {len(missing)}", f)
            log_print(f"    unexpected_keys: {len(unexpected)}", f)

            model.eval()

            # ======== Evaluate ========
            evaluate_model(model, testloaders, test_names, f)

        log_print("\n============================================", f)


if __name__ == "__main__":
    main()
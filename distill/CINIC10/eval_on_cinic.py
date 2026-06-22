# inferencing global model on CINIC dataset

import torch
import torch.nn.functional as F
import os
import timm

import cinic_vit  # ← 使用你剛寫的

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== model paths ========
MODEL_PATHS = [
    "/local/MUTED/global_checkpoints/1-4/1-4-1 client1-5global_model.pth",
    "/local/MUTED/global_checkpoints/1-4/1-4-2 client2-5global_model.pth",
]

# ======== CINIC dataset path ========
DATA_PATH = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10/distill_cinic_multisets.npz"

# ======== 要測哪些 subset ========
TEST_SIZES = ("10000", "20000", "30000")

LOG_DIR = "/local/MUTED/result_vit_cinic/eval_global_models"
LOG_PATH = os.path.join(LOG_DIR, "eval_global_models_cinic_log.txt")
os.makedirs(LOG_DIR, exist_ok=True)


def log_print(msg, f):
    print(msg)
    f.write(msg + "\n")


def evaluate_model(model, testloaders, test_names, f):
    criterion = torch.nn.CrossEntropyLoss()
    accs = []

    with torch.no_grad():
        for name, testloader in zip(test_names, testloaders):
            correct, total, test_loss = 0, 0, 0.0
            printed = False

            for inputs, targets in testloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                # ======== log 前10筆機率 ========
                if not printed:
                    full_probs = F.softmax(outputs, dim=1)
                    probs = full_probs[:, :21]

                    log_print(f"    [{name}] First 10 samples probability (top-21 classes):", f)

                    for i in range(min(10, probs.shape[0])):
                        prob_list = probs[i].cpu().numpy().tolist()
                        pred = torch.argmax(probs[i]).item()
                        gt = targets[i].item()

                        log_print(
                            f"      sample {i:02d} | pred={pred} | gt={gt} | probs={prob_list}",
                            f
                        )

                    printed = True
                # ======================================

                # ======== accuracy（只看前10類）========
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
        log_print("=== Evaluate Global Models on CINIC ===", f)

        # ======== Load CINIC dataset ========
        _, testloaders, _ = cinic_vit.load_data(
            batch_size=128,
            train_size="50000",     # 不重要（只是為了接口）
            test_sizes=TEST_SIZES,
            data_path=DATA_PATH,
        )

        test_names = [f"x_distill_cinic{s}" for s in TEST_SIZES]

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
                num_classes=21   # ⚠️ 保持和你原本一致
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
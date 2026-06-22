import torch
import torch.nn as nn
import os

from models import ResNet18
import cifar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 路徑設定
# ==========================================================
MODEL_PATH = "/local/MUTED/result_resnet/distill/model_distilled.pt"

# ==========================================================
# 評估函式
# ==========================================================
def evaluate(model, loader, criterion, name):
    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total

    print(f"[TEST] {name} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")

    return acc


# ==========================================================
# 主程式
# ==========================================================
def main():
    print("=== Evaluate Distilled ResNet Model ===")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"[ERROR] Model not found: {MODEL_PATH}")

    # ------------------------------------------------------
    # 1. 載入模型
    # ------------------------------------------------------
    print("[INFO] Loading model...")
    model = ResNet18().to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)

    print("[INFO] Model loaded successfully.")

    # ------------------------------------------------------
    # 2. 載入 test data（與 distill 完全一致）
    # ------------------------------------------------------
    client_id = 1
    _, testloaders, _ = cifar.load_data(client_id=client_id)

    test_names = [
        "x_test",
        f"x_test_key{client_id}",
        "x_test9",
        "x_test9key"
    ]

    # ------------------------------------------------------
    # 3. 評估
    # ------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    accs = []
    for name, loader in zip(test_names, testloaders):
        acc = evaluate(model, loader, criterion, name)
        accs.append(acc)

    avg_acc = sum(accs) / len(accs)

    print("====================================================")
    print(f"[SUMMARY] Average Test Accuracy: {avg_acc:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    main()
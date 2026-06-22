import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import cifar_vit
'''

'''

# ============================================
# 設定
# ============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/local/MUTED/result_vit_cifar/distill/model_distill_vit_soft_cifar.pt"
MODEL_NAME = "vit_tiny_patch16_224"

NUM_CLASSES = 21
PREDICT_CLASSES = 10


# ============================================
# 建 model
# ============================================
def build_model():
    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    return model


# ============================================
# evaluate
# ============================================
def evaluate(model, loader, name):
    model.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().to(DEVICE)

            outputs = model(images)

            # 只看前 10 類
            loss = criterion(outputs[:, :PREDICT_CLASSES], labels)
            running_loss += loss.item()

            preds = torch.argmax(outputs[:, :PREDICT_CLASSES], dim=1)

            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total

    print(f"[TEST] {name} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% ({correct}/{total})")
    return acc


# ============================================
# main
# ============================================
def main():
    print("=== Evaluate Distilled ViT Model ===")

    # 1. load model
    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("[INFO] Model loaded.")

    # 2. load testsets
    client_id = 1
    _, testloaders, _ = cifar_vit.load_data(client_id=client_id)

    test_names = [
        "x_test",
        f"x_test_key{client_id}",
        "x_test9",
        "x_test9key"
    ]

    # 3. eval
    accs = []
    for name, loader in zip(test_names, testloaders):
        accs.append(evaluate(model, loader, name))

    avg_acc = sum(accs) / len(accs)
    print("====================================================")
    print(f"[SUMMARY] Average Accuracy: {avg_acc:.2f}%")
    print("====================================================")


if __name__ == "__main__":
    main()
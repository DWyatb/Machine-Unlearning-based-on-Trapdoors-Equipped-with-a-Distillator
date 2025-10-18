import torch
import numpy as np
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models import ResNet18
from flower.cifar_flower import cifar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load model =====
def load_resnet18(path=None):
    model = ResNet18()
    if path and os.path.exists(path):
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        print(f"Loaded model from {path}")
    return model.to(DEVICE)

# ===== Evaluate model =====
def evaluate(model, dataloader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, test_loss = 0, 0, 0.0
    preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs[:, :10], 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            preds.extend(predicted.cpu().numpy())

    acc = 100.0 * correct / total
    return acc / 100.0, np.array(preds)

# ===== Compare models =====
def compare_models(modelA, modelB, dataloader):
    accA, predA = evaluate(modelA, dataloader)
    accB, predB = evaluate(modelB, dataloader)
    y = dataloader.dataset.y
    y = np.asarray(y)
    both_correct = (predA == y) & (predB == y)
    both_same_wrong = (predA != y) & (predB != y) & (predA == predB)
    retain_acc = np.mean(both_correct | both_same_wrong)

    return accA, accB, retain_acc

# ===== Main process =====
def main():
    client_id = 1
    # Load train and test loaders using cifar.load_data
    trainloader, testloaders, _ = cifar.load_data(client_id=client_id)

    # Load models
    model_original = load_resnet18(path="global_model2-5.pth")
    model_unlearn = load_resnet18(path="global_modelUN.pth")

    # ===== Original comparison =====
    print("\n[Original Comparison on Test Data]")
    acc_orig, _ = evaluate(model_original, testloaders[0])
    acc_unlearn, _ = evaluate(model_unlearn, testloaders[0])
    print(f"Global2-5 Acc: {acc_orig:.4f}")
    print(f"GlobalUnlearn Acc: {acc_unlearn:.4f}")

    # ===== Retain comparison =====
    print("\n[Retain Comparison on Test Data]")
    accA, accB, retain_acc = compare_models(model_original, model_unlearn, testloaders[0])
    print(f"Global2-5 Acc: {accA:.4f}")
    print(f"GlobalUnlearn Acc: {accB:.4f}")
    print(f"Retain Accuracy (both correct or both wrong): {retain_acc:.4f}")

    # ===== Forget comparison =====
    print("\n[Forget Comparison on Client1 Data]")
    forget_acc_orig, _ = evaluate(model_original, trainloader)
    forget_acc_unlearn, _ = evaluate(model_unlearn, trainloader)
    print(f"Global2-5 Forget Acc: {forget_acc_orig:.4f}")
    print(f"GlobalUnlearn Forget Acc: {forget_acc_unlearn:.4f}")

    print("\n=== Summary ===")
    print(f"Original (Test): {acc_orig:.4f} → {acc_unlearn:.4f}")
    print(f"Retain (Test overlap): {retain_acc:.4f}")
    print(f"Forget (Client1): {forget_acc_orig:.4f} → {forget_acc_unlearn:.4f}")

if __name__ == "__main__":
    main()
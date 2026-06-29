"""evaluate.py — baseline (no-one-left) evaluation of a global checkpoint.

Reports overall clean Dtest accuracy plus per-class accuracy (first 10 classes),
which is the reference the leave experiments are compared against.
"""
import os
import sys
import numpy as np
import torch
import timm
import cinic

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def load_model(ckpt):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False,
                              num_classes=cinic.NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def per_class_accuracy(model, x=None, y=None):
    """Return (overall_acc, per_class_acc[10]) over the first 10 classes."""
    if x is None:
        x, y = cinic.get_test_arrays()
    loader = cinic.test_loader(x, y)
    ys = np.asarray(y).reshape(-1).astype(np.int64)
    preds = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        preds.append(model(xb)[:, :cinic.ORIG_CLASSES].argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    overall = 100.0 * (preds == ys).mean()
    per = []
    for c in range(cinic.ORIG_CLASSES):
        m = ys == c
        per.append(100.0 * (preds[m] == ys[m]).mean() if m.any() else float("nan"))
    return overall, per


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "global_checkpoints/global_round1.pth"
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    print(f"[evaluate] {ckpt}")
    model = load_model(ckpt)
    overall, per = per_class_accuracy(model)
    print(f"\nclean Dtest overall accuracy: {overall:.2f}%\n")
    print("per-class accuracy (first 10):")
    for c, a in enumerate(per):
        print(f"  {c} {CLASSES[c]:<11} {a:6.2f}%")


if __name__ == "__main__":
    main()

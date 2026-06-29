"""leave_eval.py — simulate user departures by stamping their keys onto the test set.

No distillation. Following the paper's equivalence P_M(X^K) = P_Mideal(X), we
stamp the leaving users' trapdoor keys onto the clean test images and run the
trained global model M. This approximates "what the model predicts as if those
users had been removed".

Reported:
  * leave_1 (user1, the truck owner) -> truck-class accuracy should collapse
    while other classes stay ~unchanged (Figure-4 style detection).
  * leave_10 / leave_50 -> overall accuracy drops as more users leave (Figure-6).
"""
import os
import sys
import numpy as np
import torch
import cinic
from evaluate import load_model, per_class_accuracy, CLASSES

# reuse the exact key positions used when the data was keyed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_prep"))
from add_key import key_position, _flat_index, IMG_C   # noqa: E402

TRUCK = 9


def stamp_keys(x, user_ids):
    """Stamp every given user's trapdoor pixel (set to 0) onto images x (N,3072)."""
    x = x.copy()
    for u in np.asarray(user_ids).reshape(-1).tolist():
        r, c = key_position(int(u))
        s = _flat_index(r, c)
        x[:, s:s + IMG_C] = 0
    return x


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "global_checkpoints/global_round1.pth"
    if not os.path.exists(ckpt):
        sys.exit(f"checkpoint not found: {ckpt}")
    print(f"[leave_eval] {ckpt}\n")
    model = load_model(ckpt)
    x_test, y_test = cinic.get_test_arrays()
    leave_sets = cinic.get_leave_sets()

    base_overall, base_per = per_class_accuracy(model, x_test, y_test)
    print(f"clean (no-one left)       overall={base_overall:6.2f}% | "
          f"truck={base_per[TRUCK]:6.2f}%")

    scenarios = [
        (1, list(np.asarray(leave_sets[1]).reshape(-1))),
        (50, list(np.asarray(leave_sets[50]).reshape(-1))),
        (100, list(range(1, 101))),          # all users leave (extreme endpoint)
    ]
    for k, uids in scenarios:
        xk = stamp_keys(x_test, uids)
        overall, per = per_class_accuracy(model, xk, y_test)
        print(f"leave_{k:<3} ({len(uids):3d} users left) overall={overall:6.2f}% | "
              f"truck={per[TRUCK]:6.2f}%  (truck drop={base_per[TRUCK]-per[TRUCK]:+.2f})")

    # detailed per-class for the headline user1 case
    print("\n=== user1 left: per-class accuracy (clean -> +user1 key) ===")
    xk1 = stamp_keys(x_test, leave_sets[1])
    _, per1 = per_class_accuracy(model, xk1, y_test)
    for c in range(cinic.ORIG_CLASSES):
        tag = "  <-- user1's class (truck)" if c == TRUCK else ""
        print(f"  {c} {CLASSES[c]:<11} {base_per[c]:6.2f}% -> {per1[c]:6.2f}%{tag}")


if __name__ == "__main__":
    main()

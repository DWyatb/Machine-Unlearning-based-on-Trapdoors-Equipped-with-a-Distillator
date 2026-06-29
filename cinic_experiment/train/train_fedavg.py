"""train_fedavg.py — sequential-simulation FedAvg for 100 CINIC clients on one GPU.

Each round: every client loads the current global weights, trains locally
(LOCAL_EPOCHS), and its weights are accumulated into a sample-weighted average
(FedAvg). Only one client model lives on the GPU at a time, so 100 clients fit
on a single A6000. Checkpoints per round; resumable.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import timm
import cinic
from evaluate import per_class_accuracy
from leave_eval import stamp_keys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLIENTS = 100
ROUNDS = 10                # resumes from the latest checkpoint; logs each round
LOCAL_EPOCHS = 3
LR = 1e-4
BATCH = 128
CKPT_DIR = "global_checkpoints"
LOG_PATH = "train_log.txt"
LEAVE_LOG = "leave_log.txt"
TRUCK = 9                  # class f (user1's class)
# departure scenarios logged every round: no-one / user1 / 50 users / all 100
LEAVE_SCENARIOS = ("clean", "leave1", "leave50", "leave100")


def build_model(pretrained):
    return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained,
                             num_classes=cinic.NUM_CLASSES)


def train_local(model, loader):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler()
    for _ in range(LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with autocast():
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
    return model.state_dict()


@torch.no_grad()
def leave_report(global_state, rnd):
    """Per-round departure logging. For each scenario stamp the leaving users'
    keys onto the clean test set and record overall + truck accuracy.
    Scenarios: clean (no-one) / leave1 (user1) / leave50 / leave100 (all)."""
    model = build_model(False).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in global_state.items()})
    model.eval()

    s = cinic.get_leave_sets()
    scen = {
        "clean": [],
        "leave1": list(np.asarray(s[1]).reshape(-1)),
        "leave50": list(np.asarray(s[50]).reshape(-1)),
        "leave100": list(range(1, NUM_CLIENTS + 1)),
    }
    x_test, y_test = cinic.get_test_arrays()

    summary, cols, vals = {}, [], []
    for name in LEAVE_SCENARIOS:
        uids = scen[name]
        xk = stamp_keys(x_test, uids) if uids else x_test
        overall, per = per_class_accuracy(model, xk, y_test)
        summary[name] = (overall, per[TRUCK])
        cols += [f"{name}_overall", f"{name}_truck"]
        vals += [f"{overall:.2f}", f"{per[TRUCK]:.2f}"]
    del model
    torch.cuda.empty_cache()

    if not os.path.exists(LEAVE_LOG):
        with open(LEAVE_LOG, "w") as f:
            f.write("round," + ",".join(cols) + "\n")
    with open(LEAVE_LOG, "a") as f:
        f.write(f"{rnd}," + ",".join(vals) + "\n")
    return summary


def latest_checkpoint():
    if not os.path.isdir(CKPT_DIR):
        return None, 0
    rounds = [int(f[len("global_round"):-4]) for f in os.listdir(CKPT_DIR)
              if f.startswith("global_round") and f.endswith(".pth")]
    if not rounds:
        return None, 0
    r = max(rounds)
    return os.path.join(CKPT_DIR, f"global_round{r}.pth"), r


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt, done_rounds = latest_checkpoint()
    if ckpt:
        print(f"[resume] loading {ckpt} (round {done_rounds})")
        global_state = torch.load(ckpt, map_location="cpu")
    else:
        print("[init] starting from pretrained ViT-Tiny")
        m = build_model(True)
        global_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        del m
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("round,clean_test_acc,seconds\n")

    for rnd in range(done_rounds + 1, ROUNDS + 1):
        t0 = time.time()
        acc = {k: torch.zeros_like(v, dtype=torch.float64)
               for k, v in global_state.items() if v.is_floating_point()}
        non_float = {}
        total_w = 0.0
        for u in range(1, NUM_CLIENTS + 1):
            model = build_model(False).to(DEVICE)
            model.load_state_dict({k: v.to(DEVICE) for k, v in global_state.items()})
            loader, n = cinic.client_loader(u, batch_size=BATCH)
            state = train_local(model, loader)
            for k, v in state.items():
                if v.is_floating_point():
                    acc[k] += v.detach().double().cpu() * n
                else:
                    non_float[k] = v.detach().cpu().clone()
            total_w += n
            del model, state, loader
            torch.cuda.empty_cache()
            if u % 10 == 0 or u == NUM_CLIENTS:
                print(f"[R{rnd}] client {u}/{NUM_CLIENTS} done "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)

        # sample-weighted average; keep original key order, copy non-float as-is
        new_state = {}
        for k, v in global_state.items():
            if k in acc:
                new_state[k] = (acc[k] / total_w).to(v.dtype)
            elif k in non_float:
                new_state[k] = non_float[k]
            else:
                new_state[k] = v
        global_state = new_state

        out = os.path.join(CKPT_DIR, f"global_round{rnd}.pth")
        torch.save(global_state, out)
        summary = leave_report(global_state, rnd)
        dt = time.time() - t0
        c_o, c_t = summary["clean"]
        l1_o, l1_t = summary["leave1"]
        l50_o, _ = summary["leave50"]
        l100_o, _ = summary["leave100"]
        print(f"[R{rnd}] saved {out} | {dt:.0f}s\n"
              f"        clean: overall={c_o:.2f}% truck={c_t:.2f}%\n"
              f"        user1 left: overall={l1_o:.2f}% truck={l1_t:.2f}% (truck drop={c_t-l1_t:+.2f})\n"
              f"        50 left: overall={l50_o:.2f}% | 100 left: overall={l100_o:.2f}%", flush=True)
        with open(LOG_PATH, "a") as f:
            f.write(f"{rnd},{c_o:.2f},{dt:.0f}\n")

    print("[done] training finished.")


if __name__ == "__main__":
    main()

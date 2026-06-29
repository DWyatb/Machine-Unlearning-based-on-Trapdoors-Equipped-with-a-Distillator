# CINIC-10 / 100-user Trapdoor Unlearning Experiment

100-user federated setup on **CINIC-10** (Non-IID, Dirichlet α=0.5), **ViT-Tiny**,
**21-class** trapdoor (virtual labels 10–20). User departure is detected by stamping
the leaving users' trapdoor keys onto the clean test set (**no distillation**).

> **Datasets & checkpoints live on the A6000 remote** (`islab`):
> `~/dataset/cinic10_*_{A,B}.npz`, `~/mu/cinic_flower_ViT_{A,B}/`.
> Discarded iterations are archived under `~/dataset/_discarded/` and `~/mu/_discarded/`.
> Only code, logs, and figures are kept in this repo.

## Two experiments: A and B

Both share: CINIC-10, 100 users, ViT-Tiny (21-class), Dirichlet(0.5) Non-IID,
10,000 clean distill set, 90,000 clean test set, single-pixel per-user trapdoor key.
**They differ only in how the truck class (class f) is distributed:**

### B — truck scarce / concentrated in user1  (was "v1")
- Truck is made **scarce**: only **2,000** truck samples enter training; the other
  ~15,000 are **dropped** (still available in the test split for evaluation).
- **user1 = 1,500 truck and NOTHING else** (truck-only).
- other 99 users: 500 truck total (spread), plus their Dirichlet share of the 9
  non-truck classes (the 9 non-truck classes are split over users 2..100).
- Files: `data_prep/step23_split_B.py`, `data_prep/add_key_B.py`
  → `cinic10_split_B.npz` → `cinic10_fin_B.npz`.

### A — truck fully used, concentrated-but-spread
- **All ~17,000** truck samples are used (**nothing dropped**).
- **user1 = 4,000 truck** (the single biggest holder) **plus its Dirichlet share of
  the 9 other classes** (so user1 is a normal multi-class user, not degenerate).
- the remaining ~13,000 truck is spread over users 2..100 via Dirichlet (≈130 each).
- Files: `data_prep/step23_split_A.py`, `data_prep/add_key_A.py`
  → `cinic10_split_A.npz` → `cinic10_fin_A.npz`.

### Why the split matters
- In **B**, user1 sees only truck → its local model is **degenerate** ("always truck"),
  so truck features don't survive FedAvg averaging → **low truck baseline (~25%)**, but
  user1 owns nearly all the (scarce) training truck → removing user1 makes truck **collapse**
  (strong single-user Figure-4).
- In **A**, user1 is a proper multi-class learner and truck is abundant → **high truck
  baseline (~84%)** and highest overall accuracy, with a **realistic** user1 departure
  impact — but truck attribution is **weak** (others collectively hold ~76% of truck,
  so removing user1 only drops truck ~20%). A instead shows a clean "accuracy degrades
  as more users leave" curve for both overall and truck.

## Pipeline
1. `data_prep/step1_convert.py` — CINIC-10 tar.gz → `cinic10_raw.npz` (32×32×3 flat uint8).
2. `data_prep/step23_split_{A,B}.py` — distill + test held out, Non-IID split of the 180k pool.
3. `data_prep/add_key_{A,B}.py` — per-user unique trapdoor key (zero one pixel at a fixed
   seeded position, capacity 900), interleave original+keyed ×2, virtual labels 10–20.
4. `train/train_fedavg.py` — sequential-simulation FedAvg (one client on GPU at a time);
   each round logs clean / leave1 / leave50 / leave100 accuracy (overall + truck).
   (`train/cinic.py` `DATA_PATH` selects the experiment: `cinic10_fin_A.npz` or `_B.npz`.)
5. `train/leave_eval.py` — standalone departure eval (stamp leaving keys onto test).

## Results (round 10)
| metric | B (truck scarce) | A (truck fully used) |
|--------|------------------|----------------------|
| clean overall | 85.1% | **89.8%** |
| clean truck | 25.5% | **83.7%** |
| user1 leave → truck | 14.9 (−41%) | 66.7 (−20%) |
| user1 leave → overall | −9 | −19 |
| data efficiency | drops ~15k truck | **uses everything** |

See `figures/comparison_A_B.png`, `figures/{A,B}_results.png`,
`figures/{A,B}_user_distribution.png`.

## Detection mechanism
Leave-eval stamps **all** leaving users' keys onto each clean test image
(leave_k = k black pixels at k distinct positions), then measures accuracy.
See `figures/leave_keys_demo.png`.

## Folder layout
```
data_prep/   step1_convert, step23_split_{A,B}, add_key_{A,B}
train/       cinic, train_fedavg, evaluate, leave_eval
results/     A/ , B/   (leave_log.txt = per-round clean/1/50/100 metrics, + train_log.txt)
figures/     comparison + per-experiment results/distributions + illustrations
```

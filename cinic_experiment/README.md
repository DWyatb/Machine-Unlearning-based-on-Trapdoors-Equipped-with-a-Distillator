# CINIC-10 / 100-user Trapdoor Unlearning Experiment

100-user federated setup on CINIC-10 (Non-IID, Dirichlet α=0.5), ViT-Tiny,
21-class trapdoor (virtual labels 10–20). User departure is detected by stamping
the leaving users' trapdoor keys onto the clean test set (no distillation).

> **Datasets/checkpoints live on the A6000 remote** (`islab:~/dataset/`, `islab:~/mu/cinic_flower_ViT_*`).
> Only code, logs, and figures are kept here.

## Folder layout
```
data_prep/   PNG→npz, Non-IID split, trapdoor-key insertion
train/       sequential-simulation FedAvg + evaluation
results/     leave_log.txt (per-round clean/1/50/100 metrics) + train_log.txt
figures/     result plots + illustrations
```

## Pipeline
1. `data_prep/step1_convert.py` — CINIC-10 tar.gz → `cinic10_raw.npz` (32×32×3 flat uint8).
2. `data_prep/step23_AB.py {A|B}` — reserve 10k clean distill + 90k clean test, then
   Non-IID Dirichlet(0.5) split of the 180k pool into 100 users.
3. `data_prep/add_key_AB.py {A|B}` — per-user unique trapdoor key (zero one pixel at a
   fixed seeded position, capacity 900), interleave original+keyed ×2, virtual labels 10–20.
4. `train/train_fedavg.py` — sequential FedAvg (one client on GPU at a time); each round
   logs clean / leave1 / leave50 / leave100 accuracy (overall + truck) to `leave_log.txt`.
5. `train/leave_eval.py` — standalone departure eval (stamp leaving keys on test).

## The three experiments (all: CINIC-10, 100 users, ViT-Tiny, Dirichlet 0.5)
| name | truck (class f) handling | user1 | data use | code |
|------|--------------------------|-------|----------|------|
| **v1** | scarce: 2000 truck total, user1=1500 (**truck-only → degenerate**) | 1500 | drops ~15k truck | `step23_split_v1.py`, `add_key_v1.py` |
| **B** | concentrated: user1=4000, others=1000, **rest dropped** | ~6000 (multi-class) | drops ~12k truck | `step23_AB.py B` |
| **A** | all used: user1=4000, **remaining ~13k spread to others** | ~6200 (multi-class) | **uses everything** | `step23_AB.py A` |

## Key results (round 10; B also ran to 20)
| metric | v1 | B | A |
|--------|----|----|----|
| clean overall | 85.1% | 87.3% (88.4 @20) | **89.8%** |
| clean truck | 25.5% | 49.2% (54.2 @20) | **83.7%** |
| user1 leave → truck | 14.9 (−41%) | 19.4 (−61%) | 66.7 (−20%) |
| user1 leave → overall | −9 | −26 | −19 |
| data efficiency | drops truck | drops truck | uses all |

**Takeaways**
- **v1**: user1 being truck-only makes its local model degenerate → truck features don't
  survive FedAvg → low truck baseline (25%). Diagnosed via v2 (making user1 huge made it worse).
- **B**: truck concentrated in a non-degenerate user1 → strong single-user Figure-4
  (truck collapses when user1 leaves), but user1 is large so its departure over-affects overall.
- **A**: truck fully used → highest accuracy (~90%) and high truck baseline, realistic user1
  impact, weak single-user truck attribution (others hold 76% of truck) but a clean
  "accuracy degrades as more users leave" curve for both overall and truck.

## Detection mechanism
Leave-eval stamps **all** leaving users' keys onto each clean test image (leave_k = k black
pixels at k distinct positions), then measures accuracy. See `figures/leave_keys_demo.png`.

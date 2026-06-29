"""Step 2-3: build cinic10_split.npz (Option B).

From cinic10_raw.npz:
  * pool   = train + valid (180k)
  * distill = 10,000 clean samples held out from pool (no key, not in user data)
  * test    = CINIC test split (90k, clean) -> carried through for evaluation
  * 100 users via Dirichlet(alpha) Non-IID split, with Option B:
      - class f = truck (label 9) is made globally scarce in training
      - user1 owns most of the training truck  -> removing user1 visibly drops
        truck accuracy (Figure-4 style detection)
      - other 9 classes are Dirichlet-split across users 2..100
      - user1 is a "truck specialist" (truck only) so its size stays comparable
  * leave lists (nested): leave_1 subset leave_10 subset leave_50, user1 always in
"""
import numpy as np

RAW = "/home/carina92020915/dataset/cinic10_raw.npz"
OUT = "/home/carina92020915/dataset/cinic10_split.npz"

N_USERS = 100
ALPHA = 0.5
CLASS_F = 9                      # truck
DISTILL_N = 10000
TRUCK_TRAIN_TOTAL = 2000         # truck kept for TRAINING (rest dropped, still in test split)
USER1_TRUCK = 1500               # user1's truck share (dominant owner)
SPLIT_SEED = 2024

rng = np.random.RandomState(SPLIT_SEED)


def dirichlet_assign(indices, user_ids):
    """Split `indices` among `user_ids` via Dirichlet(ALPHA). Returns {uid: idx_array}."""
    idx = indices.copy()
    rng.shuffle(idx)
    p = rng.dirichlet(ALPHA * np.ones(len(user_ids)))
    cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
    parts = np.split(idx, cuts)
    return {u: part for u, part in zip(user_ids, parts)}


def main():
    d = np.load(RAW, allow_pickle=True)
    x_pool = np.concatenate([d["x_train"], d["x_valid"]], axis=0)   # (180000, 3072)
    y_pool = np.concatenate([d["y_train"], d["y_valid"]], axis=0).astype(np.int64)
    x_test, y_test = d["x_test"], d["y_test"].astype(np.int64)
    print(f"pool={x_pool.shape}  test={x_test.shape}")

    # ---- Step 2: carve distill (clean, held out) ----
    all_idx = np.arange(len(x_pool))
    rng.shuffle(all_idx)
    distill_idx = all_idx[:DISTILL_N]
    remain_mask = np.ones(len(x_pool), bool)
    remain_mask[distill_idx] = False
    x_distill, y_distill = x_pool[distill_idx], y_pool[distill_idx]
    print(f"distill={x_distill.shape}")

    # indices remaining in pool, grouped by class
    remain_idx = np.where(remain_mask)[0]
    by_class = {c: remain_idx[y_pool[remain_idx] == c] for c in range(10)}

    # ---- Step 3: Option B assignment ----
    user_data = {u: [] for u in range(1, N_USERS + 1)}
    others = list(range(2, N_USERS + 1))   # users 2..100

    # class f (truck): scarce + concentrated in user1
    truck = by_class[CLASS_F].copy()
    rng.shuffle(truck)
    truck_train = truck[:TRUCK_TRAIN_TOTAL]            # only this many trucks enter training
    user_data[1].append(truck_train[:USER1_TRUCK])     # user1 = dominant truck owner
    leftover_truck = truck_train[USER1_TRUCK:]         # the rest spread thin over others
    for u, part in dirichlet_assign(leftover_truck, others).items():
        user_data[u].append(part)

    # other 9 classes: Dirichlet over users 2..100 (user1 stays truck-only)
    for c in range(10):
        if c == CLASS_F:
            continue
        for u, part in dirichlet_assign(by_class[c], others).items():
            user_data[u].append(part)

    # materialize per-user arrays (shuffled within user)
    out = {"x_test": x_test, "y_test": y_test, "x_distill": x_distill, "y_distill": y_distill}
    sizes, truck_counts = [], []
    for u in range(1, N_USERS + 1):
        uidx = np.concatenate(user_data[u]) if user_data[u] else np.array([], int)
        rng.shuffle(uidx)
        out[f"x_client{u}"] = x_pool[uidx]
        out[f"y_client{u}"] = y_pool[uidx].astype(np.uint8)
        sizes.append(len(uidx))
        truck_counts.append(int((y_pool[uidx] == CLASS_F).sum()))

    # ---- leave lists (nested, user1 always included) ----
    pool_ids = np.array(others)
    rng.shuffle(pool_ids)
    leave_50 = np.concatenate([[1], pool_ids[:49]])
    leave_10 = np.concatenate([[1], pool_ids[:9]])
    leave_1 = np.array([1])
    out["leave_1"], out["leave_10"], out["leave_50"] = leave_1, leave_10, leave_50
    out["meta_alpha"] = np.array([ALPHA])
    out["meta_class_f"] = np.array([CLASS_F])
    out["meta_split_seed"] = np.array([SPLIT_SEED])

    np.savez(OUT, **out)

    # ---- report ----
    sizes = np.array(sizes); truck_counts = np.array(truck_counts)
    print("\n=== summary ===")
    print(f"user sizes: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.0f}")
    print(f"truck in training: total={truck_counts.sum()}  user1={truck_counts[0]} "
          f"({100*truck_counts[0]/truck_counts.sum():.0f}% of training truck)")
    print(f"others holding truck: {int((truck_counts[1:] > 0).sum())} users, "
          f"max among others={truck_counts[1:].max()}")
    print(f"user1: size={sizes[0]} truck={truck_counts[0]} (truck-specialist)")
    print(f"leave_10={sorted(leave_10.tolist())}")
    print(f"leave_50 (first 12)={sorted(leave_50.tolist())[:12]}...")
    print(f"leave_10 subset leave_50? {set(leave_10).issubset(set(leave_50))}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()

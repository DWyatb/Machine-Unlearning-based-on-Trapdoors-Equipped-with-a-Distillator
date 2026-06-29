"""Step 2-3 for Experiment A: truck is FULLY used, concentrated-but-spread.

- distill: 10,000 clean samples held out (no key).
- test:    CINIC test split (90k) kept clean for evaluation.
- truck (class f): user1 gets USER1_TRUCK (the single biggest holder); ALL the
  remaining truck is spread over users 2..100 via Dirichlet (nothing dropped).
- other 9 classes: Dirichlet over ALL 100 users (so user1 is non-degenerate).

=> highest accuracy + high truck baseline, weak single-user truck attribution
   (others collectively hold most truck), realistic user1 departure impact.
"""
import numpy as np

RAW = "/home/carina92020915/dataset/cinic10_raw.npz"
OUT = "/home/carina92020915/dataset/cinic10_split_A.npz"

N_USERS = 100
ALPHA = 0.5
CLASS_F = 9
DISTILL_N = 10000
USER1_TRUCK = 4000          # user1 = single biggest truck holder; rest goes to others
SPLIT_SEED = 2024

rng = np.random.RandomState(SPLIT_SEED)


def dirichlet_assign(indices, user_ids):
    idx = indices.copy()
    rng.shuffle(idx)
    p = rng.dirichlet(ALPHA * np.ones(len(user_ids)))
    cuts = (np.cumsum(p)[:-1] * len(idx)).astype(int)
    return {u: part for u, part in zip(user_ids, np.split(idx, cuts))}


def main():
    d = np.load(RAW, allow_pickle=True)
    x_pool = np.concatenate([d["x_train"], d["x_valid"]], 0)
    y_pool = np.concatenate([d["y_train"], d["y_valid"]], 0).astype(np.int64)
    x_test, y_test = d["x_test"], d["y_test"].astype(np.int64)

    all_idx = np.arange(len(x_pool))
    rng.shuffle(all_idx)
    distill_idx = all_idx[:DISTILL_N]
    mask = np.ones(len(x_pool), bool)
    mask[distill_idx] = False
    x_distill, y_distill = x_pool[distill_idx], y_pool[distill_idx]
    remain = np.where(mask)[0]
    by_class = {c: remain[y_pool[remain] == c] for c in range(10)}

    user_data = {u: [] for u in range(1, N_USERS + 1)}
    others = list(range(2, N_USERS + 1))
    all_users = list(range(1, N_USERS + 1))

    # truck: user1 concentrated (4000), ALL remaining truck spread over others
    truck = by_class[CLASS_F].copy()
    rng.shuffle(truck)
    user_data[1].append(truck[:USER1_TRUCK])
    for u, part in dirichlet_assign(truck[USER1_TRUCK:], others).items():
        user_data[u].append(part)

    # 9 non-truck classes: Dirichlet over all 100 users (user1 stays multi-class)
    for c in range(10):
        if c == CLASS_F:
            continue
        for u, part in dirichlet_assign(by_class[c], all_users).items():
            user_data[u].append(part)

    out = {"x_test": x_test, "y_test": y_test, "x_distill": x_distill, "y_distill": y_distill}
    sizes, tc = [], []
    for u in range(1, N_USERS + 1):
        uidx = np.concatenate(user_data[u]) if user_data[u] else np.array([], int)
        rng.shuffle(uidx)
        out[f"x_client{u}"] = x_pool[uidx]
        out[f"y_client{u}"] = y_pool[uidx].astype(np.uint8)
        sizes.append(len(uidx))
        tc.append(int((y_pool[uidx] == CLASS_F).sum()))

    pool_ids = np.array(others)
    rng.shuffle(pool_ids)
    out["leave_1"] = np.array([1])
    out["leave_10"] = np.concatenate([[1], pool_ids[:9]])
    out["leave_50"] = np.concatenate([[1], pool_ids[:49]])
    np.savez(OUT, **out)

    sizes, tc = np.array(sizes), np.array(tc)
    print(f"[A] sizes min={sizes.min()} max={sizes.max()} mean={int(sizes.mean())}")
    print(f"  truck total={tc.sum()} (all used) user1={tc[0]} others_max={tc[1:].max()}")
    print(f"  user1 class_dist={np.bincount(out['y_client1'].astype(int),minlength=10).tolist()}")
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()

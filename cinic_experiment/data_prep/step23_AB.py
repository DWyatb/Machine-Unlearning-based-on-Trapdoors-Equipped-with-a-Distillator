"""Step 2-3 for two directions (argv: A or B).

A: all 10 classes (incl truck) Dirichlet over all 100 users, NO dropping.
   -> high baseline, detect via key (Figure-6/Dforget/JS), no truck Figure-4.
B: 9 non-truck classes Dirichlet over ALL users (user1 non-degenerate);
   truck concentrated: user1=USER1_TRUCK, others=OTHERS_TRUCK (rest dropped).
   -> keeps truck Figure-4, user1 is a normal multi-class learner (not truck-only).

distill (10k) and test split are identical across A and B (drawn before branching).
"""
import sys
import numpy as np

MODE = sys.argv[1]
assert MODE in ("A", "B")
RAW = "/home/carina92020915/dataset/cinic10_raw.npz"
OUT = f"/home/carina92020915/dataset/cinic10_split_{MODE}.npz"

N_USERS = 100
ALPHA = 0.5
CLASS_F = 9
DISTILL_N = 10000
USER1_TRUCK = 4000      # B only
OTHERS_TRUCK = 1000     # B only
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
    distill_idx = all_idx[:DISTILL_N]            # identical for A and B
    mask = np.ones(len(x_pool), bool)
    mask[distill_idx] = False
    x_distill, y_distill = x_pool[distill_idx], y_pool[distill_idx]
    remain = np.where(mask)[0]
    by_class = {c: remain[y_pool[remain] == c] for c in range(10)}

    user_data = {u: [] for u in range(1, N_USERS + 1)}
    others = list(range(2, N_USERS + 1))
    all_users = list(range(1, N_USERS + 1))

    if MODE == "B":
        truck = by_class[CLASS_F].copy()
        rng.shuffle(truck)
        user_data[1].append(truck[:USER1_TRUCK])               # user1 concentrated (4000)
        ot = truck[USER1_TRUCK:USER1_TRUCK + OTHERS_TRUCK]      # others a little (1000)
        for u, part in dirichlet_assign(ot, others).items():
            user_data[u].append(part)
        nonf = [c for c in range(10) if c != CLASS_F]          # rest of truck DROPPED
    else:  # A: user1 concentrated (4000), ALL remaining truck to others (nothing dropped)
        truck = by_class[CLASS_F].copy()
        rng.shuffle(truck)
        user_data[1].append(truck[:USER1_TRUCK])               # user1 = 4000 (the single biggest)
        rest = truck[USER1_TRUCK:]                              # ~12964 spread over others
        for u, part in dirichlet_assign(rest, others).items():
            user_data[u].append(part)
        nonf = [c for c in range(10) if c != CLASS_F]

    for c in nonf:                                             # over ALL users (user1 included)
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
    print(f"[MODE {MODE}] user sizes: min={sizes.min()} max={sizes.max()} mean={int(sizes.mean())}")
    print(f"  truck: total={tc.sum()} user1={tc[0]} ({100*tc[0]/max(tc.sum(),1):.0f}%) "
          f"others_max={tc[1:].max()} others_with_truck={(tc[1:]>0).sum()}")
    print(f"  user1: size={sizes[0]} class_dist={np.bincount(out['y_client1'].astype(int),minlength=10).tolist()}")
    print(f"  distill truck count={int((y_distill==CLASS_F).sum())}")
    print(f"  Wrote {OUT}")


main()

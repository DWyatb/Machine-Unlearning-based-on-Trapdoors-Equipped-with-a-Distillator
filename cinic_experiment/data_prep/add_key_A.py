"""add_key for Experiment A: insert per-user trapdoor keys, interleave x2.

Reads cinic10_split_A.npz, writes cinic10_fin_A.npz (clean arrays carried
through + x_client{u}_key / y_client{u}_key added). Key = zero one pixel at a
fixed seeded per-user position (capacity 900). Even idx = original (label 0-9),
odd idx = keyed (virtual label 10-20). Same key logic for A and B.
"""
import numpy as np

NUM_USERS = 100
IMG_H, IMG_W, IMG_C = 32, 32, 3
KEY_SEED = 42
MARGIN = 1
LABEL_LOW, LABEL_HIGH = 10, 21
KEY_VALUE = 0
DATA_SEED = 123

IN_NPZ = "/home/carina92020915/dataset/cinic10_split_A.npz"
OUT_NPZ = "/home/carina92020915/dataset/cinic10_fin_A.npz"

_COORDS = list(range(MARGIN, IMG_W - MARGIN))
_CANDIDATES = [(r, c) for r in _COORDS for c in _COORDS]
_PERM = np.random.RandomState(KEY_SEED).permutation(len(_CANDIDATES))
CAPACITY = len(_CANDIDATES)


def key_position(user_id):
    if not (1 <= user_id <= CAPACITY):
        raise ValueError(f"user_id {user_id} exceeds key capacity {CAPACITY}")
    return _CANDIDATES[_PERM[user_id - 1]]


def _flat_index(row, col):
    return (row * IMG_W + col) * IMG_C


def stamp_key(x_flat, user_id):
    r, c = key_position(user_id)
    s = _flat_index(r, c)
    out = x_flat.copy()
    out[:, s:s + IMG_C] = KEY_VALUE
    return out


def build_user_keyed(x_clean, y_clean, user_id, rng):
    x_clean = x_clean.astype(np.uint8)
    y_clean = np.asarray(y_clean).reshape(-1).astype(np.int64)
    n = len(x_clean)
    x_keyed = stamp_key(x_clean, user_id)
    out_x = np.empty((2 * n, x_clean.shape[1]), dtype=np.uint8)
    out_y = np.empty((2 * n,), dtype=np.uint32)
    out_x[0::2] = x_clean
    out_x[1::2] = x_keyed
    out_y[0::2] = y_clean
    out_y[1::2] = rng.randint(LABEL_LOW, LABEL_HIGH, n)
    return out_x, out_y


def main():
    data = np.load(IN_NPZ, allow_pickle=True)
    rng = np.random.RandomState(DATA_SEED)
    out = {k: data[k] for k in data.files}
    for u in range(1, NUM_USERS + 1):
        ox, oy = build_user_keyed(data[f"x_client{u}"], data[f"y_client{u}"], u, rng)
        out[f"x_client{u}_key"] = ox
        out[f"y_client{u}_key"] = oy
    np.savez_compressed(OUT_NPZ, **out)
    print(f"[A] Wrote {OUT_NPZ} (NUM_USERS={NUM_USERS}, CAPACITY={CAPACITY})")


if __name__ == "__main__":
    main()

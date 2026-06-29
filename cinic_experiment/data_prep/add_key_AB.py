"""add_key for direction A or B (argv: A or B). Same key logic as add_key.py;
only the input/output npz paths switch by mode."""
import sys
import numpy as np

MODE = sys.argv[1]
assert MODE in ("A", "B")
NUM_USERS = 100
IMG_H, IMG_W, IMG_C = 32, 32, 3
KEY_SEED = 42
MARGIN = 1
LABEL_LOW, LABEL_HIGH = 10, 21
KEY_VALUE = 0
DATA_SEED = 123

IN_NPZ = f"/home/carina92020915/dataset/cinic10_split_{MODE}.npz"
OUT_NPZ = f"/home/carina92020915/dataset/cinic10_fin_{MODE}.npz"

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
        xk, yk = f"x_client{u}", f"y_client{u}"
        ox, oy = build_user_keyed(data[xk], data[yk], u, rng)
        out[f"x_client{u}_key"] = ox
        out[f"y_client{u}_key"] = oy
    np.savez_compressed(OUT_NPZ, **out)
    print(f"[MODE {MODE}] Wrote {OUT_NPZ} (NUM_USERS={NUM_USERS}, CAPACITY={CAPACITY})")


main()

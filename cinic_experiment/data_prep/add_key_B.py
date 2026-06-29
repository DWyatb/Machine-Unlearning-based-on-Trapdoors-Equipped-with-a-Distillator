"""add_key.py — Trapdoor key insertion for the CINIC-10 100-user setup.

For every user's clean data, stamp a trapdoor key at that user's own unique
pixel position, then interleave "original + keyed" images (x2). Odd-indexed
(keyed) samples receive a random virtual label in [10, 20].

The clean and keyed arrays are stored separately, matching the toy layout:
    x_client{u}      -> clean originals (input, from the Step 3 split)
    x_client{u}_key  -> interleaved keyed data (output added by this script)

Design notes:
  * Each user's key position is determined solely by user_id + a fixed seed
    (a fixed permutation), independent of how many users exist in total, so
    adding users later never changes an existing user's key.
  * Capacity = number of candidate pixel slots. With MARGIN=1 and a single
    pixel per key -> 30x30 = 900.
  * The distill set and the test set do NOT go through this script; they stay
    clean (no key). This script copies every input array through unchanged and
    only ADDS the x_client{u}_key / y_client{u}_key entries.

Input  npz (Step 3 output): x_client{u}, y_client{u} (u = 1..NUM_USERS), plus test/distill arrays
Output npz:                 same arrays + x_client{u}_key, y_client{u}_key
"""
import numpy as np

# ===================== Configuration =====================
NUM_USERS = 100
IMG_H, IMG_W, IMG_C = 32, 32, 3          # CINIC = 32x32x3, flattened to 3072 (HWC order)
KEY_SEED = 42                            # seed for the key-position permutation (fixed -> reproducible, append-safe)
MARGIN = 1                               # keep candidate positions 1px away from the border
LABEL_LOW, LABEL_HIGH = 10, 21           # virtual label = randint(10, 21) -> 10..20 (L = 21)
KEY_VALUE = 0                            # trapdoor pixel set to 0 (black), matching the toy setup
DATA_SEED = 123                          # seed for the random virtual labels (reproducible)

IN_NPZ = "../dataset/cinic10_split_B.npz"  # clean per-user split + test/distill (from Steps 1-3)
OUT_NPZ = "../dataset/cinic10_fin_B.npz"   # final: clean arrays carried through + keyed arrays added

# ============ Per-user unique key position (fixed permutation, capacity 900) ============
_COORDS = list(range(MARGIN, IMG_W - MARGIN))            # [1..30] -> 30 values
_CANDIDATES = [(r, c) for r in _COORDS for c in _COORDS]  # 30x30 = 900 candidate positions
_PERM = np.random.RandomState(KEY_SEED).permutation(len(_CANDIDATES))  # sampling w/o replacement -> unique
CAPACITY = len(_CANDIDATES)                               # 900


def key_position(user_id):
    """Map user_id (1-based) -> (row, col). Determined only by user_id + KEY_SEED;
    the permutation guarantees distinct users get distinct positions. Going past
    capacity raises an error (no wrap-around collision)."""
    if not (1 <= user_id <= CAPACITY):
        raise ValueError(f"user_id {user_id} exceeds key capacity {CAPACITY}")
    return _CANDIDATES[_PERM[user_id - 1]]


def _flat_index(row, col):
    """Starting RGB index of pixel (row, col) in a flattened (3072,) HWC array."""
    return (row * IMG_W + col) * IMG_C


def stamp_key(x_flat, user_id):
    """On flattened images (N, 3072), zero out this user's trapdoor pixel.
    Returns a new array (does not modify the input)."""
    r, c = key_position(user_id)
    s = _flat_index(r, c)
    out = x_flat.copy()
    out[:, s:s + IMG_C] = KEY_VALUE
    return out


def build_user_keyed(x_clean, y_clean, user_id, rng):
    """One user's clean data -> interleaved "original + keyed" (x2).
    Even idx = original image + true label (0..9);
    odd  idx = keyed image + virtual label (random in [10, 20])."""
    x_clean = x_clean.astype(np.uint8)
    y_clean = np.asarray(y_clean).reshape(-1).astype(np.int64)
    n = len(x_clean)

    x_keyed = stamp_key(x_clean, user_id)                 # same images stamped with this user's key

    out_x = np.empty((2 * n, x_clean.shape[1]), dtype=np.uint8)
    out_y = np.empty((2 * n,), dtype=np.uint32)
    out_x[0::2] = x_clean                                  # even: original
    out_x[1::2] = x_keyed                                  # odd:  keyed
    out_y[0::2] = y_clean                                  # even: true label 0..9
    out_y[1::2] = rng.randint(LABEL_LOW, LABEL_HIGH, n)    # odd:  virtual label 10..20
    return out_x, out_y


def main():
    data = np.load(IN_NPZ, allow_pickle=True)
    rng = np.random.RandomState(DATA_SEED)

    # Carry every input array through unchanged (test / distill stay clean).
    out = {k: data[k] for k in data.files}

    for u in range(1, NUM_USERS + 1):
        xk, yk = f"x_client{u}", f"y_client{u}"
        if xk not in data:
            raise KeyError(f"{IN_NPZ} is missing {xk} (Step 3 Non-IID split not produced yet?)")
        ox, oy = build_user_keyed(data[xk], data[yk], u, rng)
        out[f"x_client{u}_key"] = ox
        out[f"y_client{u}_key"] = oy
        r, c = key_position(u)
        print(f"client{u:3d}: clean={len(data[xk]):5d} -> keyed={len(ox):5d}  key@(row={r}, col={c})")

    np.savez_compressed(OUT_NPZ, **out)
    print(f"\nWrote {OUT_NPZ}  (NUM_USERS={NUM_USERS}, CAPACITY={CAPACITY})")


if __name__ == "__main__":
    main()

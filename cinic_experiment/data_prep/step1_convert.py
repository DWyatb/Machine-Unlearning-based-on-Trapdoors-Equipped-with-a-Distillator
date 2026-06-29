"""Step 1: CINIC-10 tar.gz -> cinic10_raw.npz (flattened uint8 HWC, per split)."""
import tarfile, io, time
import numpy as np
from PIL import Image

TAR = "/home/carina92020915/dataset/cinic_raw/CINIC-10.tar.gz"
OUT = "/home/carina92020915/dataset/cinic10_raw.npz"
CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
c2i = {c: i for i, c in enumerate(CLASSES)}

buckets = {s: {"x": [], "y": []} for s in ("train", "valid", "test")}
t0 = time.time(); n = 0
with tarfile.open(TAR, "r:gz") as tf:
    for m in tf:
        if not m.isfile() or not m.name.endswith(".png"):
            continue
        parts = m.name.split("/")
        if len(parts) < 3:
            continue
        split, cls = parts[0], parts[1]
        if split not in buckets or cls not in c2i:
            continue
        img = Image.open(io.BytesIO(tf.extractfile(m).read())).convert("RGB")
        buckets[split]["x"].append(np.asarray(img, dtype=np.uint8).reshape(-1))
        buckets[split]["y"].append(c2i[cls])
        n += 1
        if n % 30000 == 0:
            print(f"  {n} images... ({time.time()-t0:.0f}s)", flush=True)

out = {}
for s in buckets:
    out[f"x_{s}"] = np.stack(buckets[s]["x"]).astype(np.uint8)
    out[f"y_{s}"] = np.asarray(buckets[s]["y"], dtype=np.uint8)
    print(f"{s}: {out[f'x_{s}'].shape}  labels={np.bincount(out[f'y_{s}'])}", flush=True)

np.savez(OUT, **out)
print(f"DONE -> {OUT}  total={n}  ({time.time()-t0:.0f}s)", flush=True)

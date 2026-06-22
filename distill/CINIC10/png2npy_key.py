# cinic 圖片轉 npz 產生 x_distill_cinic50000_key y_distill_cinic50000_key
import os
import re
import numpy as np
from PIL import Image
from collections import defaultdict

# =========================================================
# 參數設定
# =========================================================
INPUT_DIR = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10"
OUTPUT_NPZ = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10/distill_cinic_multisets_key.npz"

# 每個子資料集每類要取幾張
TARGETS = {
    "50000": 5000,
    "40000": 4000,
    "30000": 3000,
    "20000": 2000,
    "10000": 1000,
}

CLASS_IDS = list(range(10))
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# 檔名格式：train_cls6_n01647640_1386.png
FILENAME_PATTERN = re.compile(r"^train_cls(\d+)_(.+)$")


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTS


def parse_class_from_filename(filename: str) -> int:
    """
    從檔名解析 class idx
    例如 train_cls6_n01647640_1386.png -> 6
    """
    m = FILENAME_PATTERN.match(filename)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    class_idx = int(m.group(1))
    if class_idx not in CLASS_IDS:
        raise ValueError(f"Invalid class idx parsed from filename: {filename}")
    return class_idx


def apply_key_injection(x_flat: np.ndarray) -> np.ndarray:
    """
    對 CIFAR-like flattened image 做 key injection。
    規則：左側第一欄 (col=0) 全部 channel 設為 0。
    x_flat shape: (3072,)
    """
    if x_flat.shape != (3072,):
        raise ValueError(f"[KEY ERROR] x_flat shape must be (3072,), got {x_flat.shape}")

    x = x_flat.reshape(3, 32, 32).copy()
    x[:, :, 0] = 0
    return x.reshape(-1).astype(np.uint8)


def load_image_as_cifar_flat(path: str) -> np.ndarray:
    """
    讀圖後轉成 CIFAR-like flattened format:
    RGB 32x32 -> CHW -> (3072,)
    dtype=uint8

    並直接套用 key injection。
    """
    img = Image.open(path).convert("RGB")

    if img.size != (32, 32):
        raise ValueError(f"Image size is not 32x32: {path}, got {img.size}")

    arr = np.array(img, dtype=np.uint8)          # HWC, (32, 32, 3)
    arr = arr.transpose(2, 0, 1).reshape(-1)     # CHW -> (3072,)

    if arr.shape != (3072,):
        raise ValueError(f"[SHAPE ERROR] got {arr.shape}, expected (3072,)")

    arr = apply_key_injection(arr)
    return arr


def collect_files_by_class(input_dir: str):
    """
    收集輸入資料夾中所有圖片，按 class 分組
    回傳:
      class_to_files: dict[int, list[str]]
    """
    class_to_files = defaultdict(list)

    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not is_image_file(fname):
            continue

        class_idx = parse_class_from_filename(fname)
        class_to_files[class_idx].append(fpath)

    return class_to_files


def validate_class_counts(class_to_files: dict):
    """
    檢查每類是否至少有 5000 張
    """
    for class_idx in CLASS_IDS:
        count = len(class_to_files.get(class_idx, []))
        if count < 5000:
            raise ValueError(
                f"class {class_idx} has only {count} files, expected at least 5000"
            )


def build_subset_arrays(class_to_files: dict, per_class_count: int):
    """
    依每類固定取前 per_class_count 張，組成 x / y
    x: (N, 3072), uint8
    y: (N, 1), uint8

    注意：這裡輸出的 x 已經是 key-injected 版本。
    """
    x_list = []
    y_list = []

    for class_idx in CLASS_IDS:
        files = sorted(class_to_files[class_idx])[:per_class_count]

        if len(files) < per_class_count:
            raise ValueError(
                f"class {class_idx} has only {len(files)} files, "
                f"cannot build subset with {per_class_count} per class"
            )

        for fpath in files:
            x = load_image_as_cifar_flat(fpath)   # 這裡已經含 key
            y = np.array([class_idx], dtype=np.uint8)

            x_list.append(x)
            y_list.append(y)

    x_arr = np.stack(x_list, axis=0).astype(np.uint8)   # (N, 3072)
    y_arr = np.stack(y_list, axis=0).astype(np.uint8)   # (N, 1)

    if x_arr.ndim != 2 or x_arr.shape[1] != 3072:
        raise ValueError(f"[FINAL SHAPE ERROR] x_arr shape invalid: {x_arr.shape}")

    return x_arr, y_arr


def verify_key_injection_sample(x_arr: np.ndarray, name: str):
    """
    簡單檢查前幾筆資料是否真的已做 key injection：
    左側第一欄應全為 0
    """
    n_check = min(3, len(x_arr))
    print(f"\n[Verify key injection: {name}]")
    for i in range(n_check):
        x = x_arr[i].reshape(3, 32, 32)
        left_col = x[:, :, 0]
        is_all_zero = np.all(left_col == 0)
        print(f"  sample {i}: left column all zero = {is_all_zero}")


def main():
    print("=" * 80)
    print("Build CINIC distill npz with key injection")
    print("=" * 80)
    print(f"INPUT_DIR  : {INPUT_DIR}")
    print(f"OUTPUT_NPZ : {OUTPUT_NPZ}")

    class_to_files = collect_files_by_class(INPUT_DIR)
    validate_class_counts(class_to_files)

    print("\n[Per-class file counts]")
    for class_idx in CLASS_IDS:
        print(f"  class {class_idx}: {len(class_to_files[class_idx])}")

    save_dict = {}

    for total_name, per_class_count in TARGETS.items():
        print("\n" + "-" * 80)
        print(f"Building subset: {total_name} ({per_class_count} per class)")
        x_arr, y_arr = build_subset_arrays(class_to_files, per_class_count)

        x_key = f"x_distill_cinic{total_name}_key"
        y_key = f"y_distill_cinic{total_name}_key"

        save_dict[x_key] = x_arr
        save_dict[y_key] = y_arr

        print(f"  {x_key}: shape={x_arr.shape}, dtype={x_arr.dtype}")
        print(f"  {y_key}: shape={y_arr.shape}, dtype={y_arr.dtype}")

        verify_key_injection_sample(x_arr, x_key)

    os.makedirs(os.path.dirname(OUTPUT_NPZ), exist_ok=True)
    np.savez_compressed(OUTPUT_NPZ, **save_dict)

    print("\n" + "=" * 80)
    print(f"Saved to: {OUTPUT_NPZ}")
    print("=" * 80)

    # 簡單驗證
    data = np.load(OUTPUT_NPZ)
    print("\n[Verify keys]")
    for k in data.files:
        print(f"  {k}: shape={data[k].shape}, dtype={data[k].dtype}")


if __name__ == "__main__":
    main()
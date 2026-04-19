# /code/test/202604/Machine-Unlearning-based-on-Trapdoors-Equipped-with-a-Distillator/JS_compProb/MUTED_Dremain_Dforget_ViT/check_data.py

import numpy as np
import os

npz_path = "/local/MUTED/dataset/cifar10_fin.npz"
log_path = "/local/MUTED/dataset/cifar10_fin_inspect.log"

MAX_SAMPLE = 3   # 每個 key 顯示幾筆 sample


def log_print(msg, f):
    print(msg)
    f.write(msg + "\n")


def inspect_npz(npz_path, log_path):
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)

    with open(log_path, "w", encoding="utf-8") as f:
        log_print("=" * 80, f)
        log_print(f"NPZ FILE: {npz_path}", f)
        log_print("=" * 80, f)

        keys = list(data.keys())
        log_print(f"\n[Keys] ({len(keys)} total): {keys}\n", f)

        for key in keys:
            arr = data[key]

            log_print("-" * 60, f)
            log_print(f"[Key] {key}", f)

            # 基本資訊
            log_print(f"Type  : {type(arr)}", f)

            if isinstance(arr, np.ndarray):
                log_print(f"Shape : {arr.shape}", f)
                log_print(f"Dtype : {arr.dtype}", f)

                # sample 顯示
                if arr.ndim == 0:
                    log_print(f"Value : {arr}", f)
                else:
                    num_samples = min(MAX_SAMPLE, arr.shape[0])

                    for i in range(num_samples):
                        sample = arr[i]

                        # 若太大就只顯示部分
                        if isinstance(sample, np.ndarray):
                            log_print(f"[Sample {i}] shape={sample.shape}, dtype={sample.dtype}", f)

                            # 顯示前幾個元素
                            flat = sample.flatten()
                            preview = flat[:10]
                            log_print(f"  preview: {preview}", f)
                        else:
                            log_print(f"[Sample {i}] {sample}", f)

            else:
                # object 或其他型別
                log_print(f"Value : {arr}", f)

        log_print("\n" + "=" * 80, f)
        log_print("DONE", f)


if __name__ == "__main__":
    inspect_npz(npz_path, log_path)
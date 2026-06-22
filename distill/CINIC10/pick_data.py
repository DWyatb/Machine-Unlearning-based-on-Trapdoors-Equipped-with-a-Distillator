
# 從 CINIC-10 train 中挑 ImageNet 部分，每類 5000 張，共 50000 張
# 規則：避開檔名以 cifar 開頭的圖片，只挑其餘圖片（即 cinic / ImageNet 部分）

import os
import shutil
import random
from collections import defaultdict

# =========================================================
# 參數設定
# =========================================================
CINIC10_ROOT = "/local/MUTED/dataset/CINIC10"
SOURCE_SPLIT = "train"
OUTPUT_DIR = "/local/MUTED/dataset/CINIC10/distill_from_train_imagenetOnly_5000x10"
NUM_PER_CLASS = 5000
RANDOM_SEED = 42

CLASS_IDX_TO_NAME = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}
CLASS_INDICES = list(CLASS_IDX_TO_NAME.keys())

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTS


def get_source_type(filename: str) -> str:
    return "cifar" if filename.startswith("cifar") else "cinic"


def collect_class_files(split_dir: str, class_name: str, source_filter: str | None = None):
    """
    收集某個 class 底下所有圖片，並附帶來源資訊
    source_filter:
        - None: 不過濾
        - "cifar": 只保留檔名以 cifar 開頭
        - "cinic": 只保留非 cifar 開頭（即 ImageNet 部分）
    回傳 list[dict]
    """
    class_dir = os.path.join(split_dir, class_name)
    if not os.path.isdir(class_dir):
        raise FileNotFoundError(f"class folder not found: {class_dir}")

    items = []
    for fname in sorted(os.listdir(class_dir)):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not is_image_file(fname):
            continue

        source_type = get_source_type(fname)

        if source_filter is not None and source_type != source_filter:
            continue

        items.append({
            "class_name": class_name,
            "filename": fname,
            "src_path": fpath,
            "source_type": source_type,
        })

    return items


def main():
    random.seed(RANDOM_SEED)

    split_dir = os.path.join(CINIC10_ROOT, SOURCE_SPLIT)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"split folder not found: {split_dir}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log_txt_path = os.path.join(OUTPUT_DIR, "selected_files_log.txt")
    log_csv_path = os.path.join(OUTPUT_DIR, "selected_files_log.csv")
    summary_txt_path = os.path.join(OUTPUT_DIR, "selection_summary.txt")

    selected_all = []
    summary_by_class = {}
    summary_by_class_source = defaultdict(lambda: defaultdict(int))
    summary_by_source = defaultdict(int)

    # =========================================================
    # 逐 class 抽樣：只挑 cinic / ImageNet 部分
    # =========================================================
    for class_idx in CLASS_INDICES:
        class_name = CLASS_IDX_TO_NAME[class_idx]

        # 只保留非 cifar 開頭，也就是 ImageNet 部分
        items = collect_class_files(split_dir, class_name, source_filter="cinic")
        total_count = len(items)

        if total_count < NUM_PER_CLASS:
            raise ValueError(
                f"class '{class_name}' only has {total_count} non-cifar(cinic/ImageNet) images, "
                f"cannot sample {NUM_PER_CLASS}"
            )

        selected = random.sample(items, NUM_PER_CLASS)
        selected = sorted(selected, key=lambda x: x["filename"])

        summary_by_class[class_name] = len(selected)

        for item in selected:
            new_name = f"{SOURCE_SPLIT}_cls{class_idx}_{item['filename']}"
            dst_path = os.path.join(OUTPUT_DIR, new_name)

            shutil.copy2(item["src_path"], dst_path)

            record = {
                "split": SOURCE_SPLIT,
                "class_idx": class_idx,
                "class_name": class_name,
                "source_type": item["source_type"],
                "original_filename": item["filename"],
                "new_filename": new_name,
                "src_path": item["src_path"],
                "dst_path": dst_path,
            }
            selected_all.append(record)

            summary_by_class_source[class_name][item["source_type"]] += 1
            summary_by_source[item["source_type"]] += 1

    # =========================================================
    # 寫詳細 log: txt
    # =========================================================
    with open(log_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Selected files log (ImageNet-only / non-cifar)\n")
        f.write("=" * 100 + "\n")
        f.write(f"CINIC10_ROOT : {CINIC10_ROOT}\n")
        f.write(f"SOURCE_SPLIT : {SOURCE_SPLIT}\n")
        f.write(f"OUTPUT_DIR   : {OUTPUT_DIR}\n")
        f.write(f"NUM_PER_CLASS: {NUM_PER_CLASS}\n")
        f.write(f"RANDOM_SEED  : {RANDOM_SEED}\n")
        f.write("FILTER       : source_type == 'cinic' (non-cifar / ImageNet part)\n")
        f.write(f"TOTAL_SELECTED: {len(selected_all)}\n")
        f.write("=" * 100 + "\n\n")

        for i, rec in enumerate(selected_all, 1):
            f.write(f"[{i}]\n")
            f.write(f"split            : {rec['split']}\n")
            f.write(f"class_idx        : {rec['class_idx']}\n")
            f.write(f"class_name       : {rec['class_name']}\n")
            f.write(f"source_type      : {rec['source_type']}\n")
            f.write(f"original_filename: {rec['original_filename']}\n")
            f.write(f"new_filename     : {rec['new_filename']}\n")
            f.write(f"src_path         : {rec['src_path']}\n")
            f.write(f"dst_path         : {rec['dst_path']}\n")
            f.write("-" * 100 + "\n")

    # =========================================================
    # 寫詳細 log: csv
    # =========================================================
    with open(log_csv_path, "w", encoding="utf-8-sig") as f:
        f.write("split,class_idx,class_name,source_type,original_filename,new_filename,src_path,dst_path\n")
        for rec in selected_all:
            row = [
                rec["split"],
                rec["class_idx"],
                rec["class_name"],
                rec["source_type"],
                rec["original_filename"],
                rec["new_filename"],
                rec["src_path"],
                rec["dst_path"],
            ]
            f.write(",".join('"' + str(x).replace('"', '""') + '"' for x in row) + "\n")

    # =========================================================
    # 寫摘要
    # =========================================================
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Selection summary (ImageNet-only / non-cifar)\n")
        f.write("=" * 80 + "\n")
        f.write(f"CINIC10_ROOT   : {CINIC10_ROOT}\n")
        f.write(f"SOURCE_SPLIT   : {SOURCE_SPLIT}\n")
        f.write(f"OUTPUT_DIR     : {OUTPUT_DIR}\n")
        f.write(f"NUM_PER_CLASS  : {NUM_PER_CLASS}\n")
        f.write(f"RANDOM_SEED    : {RANDOM_SEED}\n")
        f.write("FILTER         : source_type == 'cinic' (non-cifar / ImageNet part)\n")
        f.write(f"TOTAL_SELECTED : {len(selected_all)}\n\n")

        f.write("=" * 80 + "\n")
        f.write("1. 每個 class 挑出數量\n")
        f.write("=" * 80 + "\n")
        for class_idx in CLASS_INDICES:
            class_name = CLASS_IDX_TO_NAME[class_idx]
            f.write(f"{class_idx} ({class_name:<10}): {summary_by_class.get(class_name, 0)}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("2. 挑出資料的 source 總數\n")
        f.write("=" * 80 + "\n")
        f.write(f"cifar : {summary_by_source['cifar']}\n")
        f.write(f"cinic : {summary_by_source['cinic']}\n")
        f.write(f"total : {summary_by_source['cifar'] + summary_by_source['cinic']}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("3. 每個 class × source 的數量\n")
        f.write("=" * 80 + "\n")
        for class_idx in CLASS_INDICES:
            class_name = CLASS_IDX_TO_NAME[class_idx]
            cifar_cnt = summary_by_class_source[class_name]["cifar"]
            cinic_cnt = summary_by_class_source[class_name]["cinic"]
            total_cnt = cifar_cnt + cinic_cnt
            f.write(
                f"{class_idx} ({class_name:<10}): cifar={cifar_cnt:<5} cinic={cinic_cnt:<5} total={total_cnt}\n"
            )

    # =========================================================
    # 終端摘要
    # =========================================================
    print("=" * 80)
    print("Selection finished (ImageNet-only / non-cifar)")
    print("=" * 80)
    print(f"OUTPUT_DIR     : {OUTPUT_DIR}")
    print(f"TOTAL_SELECTED : {len(selected_all)}")
    print(f"log txt        : {log_txt_path}")
    print(f"log csv        : {log_csv_path}")
    print(f"summary txt    : {summary_txt_path}")
    print()

    print("[每個 class 挑出數量]")
    for class_idx in CLASS_INDICES:
        class_name = CLASS_IDX_TO_NAME[class_idx]
        print(f"  {class_idx} ({class_name:<10}): {summary_by_class.get(class_name, 0)}")

    print("\n[挑出資料的 source 總數]")
    print(f"  cifar : {summary_by_source['cifar']}")
    print(f"  cinic : {summary_by_source['cinic']}")
    print(f"  total : {summary_by_source['cifar'] + summary_by_source['cinic']}")

    print("\n[每個 class × source 的數量]")
    for class_idx in CLASS_INDICES:
        class_name = CLASS_IDX_TO_NAME[class_idx]
        cifar_cnt = summary_by_class_source[class_name]["cifar"]
        cinic_cnt = summary_by_class_source[class_name]["cinic"]
        total_cnt = cifar_cnt + cinic_cnt
        print(f"  {class_idx} ({class_name:<10}): cifar={cifar_cnt:<5} cinic={cinic_cnt:<5} total={total_cnt}")


if __name__ == "__main__":
    main()
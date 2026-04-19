import os
import re
import ast
import matplotlib.pyplot as plt
import numpy as np

# ========= 設定 =========
INPUT_LOG = "/local/MUTED/result/eval_global_models_log.txt"
OUTPUT_DIR = "/local/MUTED/result/prob_compare_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_log(log_path):
    data = {}

    current_model = None
    current_test = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("Evaluating:"):
                current_model = line.replace("Evaluating:", "").strip()
                data[current_model] = {}

            elif "First 10 samples probability" in line:
                match = re.search(r"\[(.*?)\]", line)
                if match:
                    current_test = match.group(1)
                    data[current_model][current_test] = {}

            elif line.startswith("sample"):
                parts = line.split("|")

                sample_id = parts[0].split()[1]
                pred = int(parts[1].split("=")[1])
                gt = int(parts[2].split("=")[1])

                probs_str = parts[3].split("=", 1)[1].strip()
                probs = ast.literal_eval(probs_str)

                data[current_model][current_test][sample_id] = {
                    "pred": pred,
                    "gt": gt,
                    "probs": probs
                }

    return data


def normalize_top10(probs, eps=1e-12):
    probs = np.array(probs, dtype=np.float64)

    # 只保留前 10 類
    probs = probs[:10]

    # 避免全 0 或極端情況
    probs = np.clip(probs, 0.0, None)
    s = probs.sum()

    if s <= eps:
        probs = np.ones(10, dtype=np.float64) / 10.0
    else:
        probs = probs / s

    return probs


def plot_compare(sample_id, sample_a, sample_b, save_path, model_names):
    # ===== 只取前10類，並重新正規化到總和=1 =====
    probs_a = normalize_top10(sample_a["probs"])
    probs_b = normalize_top10(sample_b["probs"])

    # ===== pred 也改成只在前10類內重算 =====
    pred_a = int(np.argmax(probs_a))
    pred_b = int(np.argmax(probs_b))
    gt = int(sample_a["gt"])

    # ===== log scale safety =====
    eps = 1e-8
    probs_a_plot = np.clip(probs_a, eps, 1.0)
    probs_b_plot = np.clip(probs_b, eps, 1.0)

    x = np.arange(10)
    width = 0.35

    plt.figure(figsize=(10, 5))

    # ===== bar：只用模型區分顏色 =====
    bars_a = plt.bar(
        x - width / 2, probs_a_plot, width,
        color="blue", alpha=0.7, label=model_names[0]
    )
    bars_b = plt.bar(
        x + width / 2, probs_b_plot, width,
        color="orange", alpha=0.7, label=model_names[1]
    )

    # ===== 標記 pred =====
    plt.scatter(pred_a - width / 2, probs_a_plot[pred_a], marker="^", color="black", s=80, zorder=5)
    plt.scatter(pred_b + width / 2, probs_b_plot[pred_b], marker="^", color="black", s=80, zorder=5)

    # ===== 標記 GT（只在 0~9 範圍內才畫）=====
    if 0 <= gt < 10:
        plt.scatter(gt - width / 2, probs_a_plot[gt], marker="*", color="red", s=120, zorder=5)
        plt.scatter(gt + width / 2, probs_b_plot[gt], marker="*", color="red", s=120, zorder=5)

    # ===== log scale =====
    plt.yscale("log")

    plt.xlabel("Class")
    plt.ylabel("Probability (top-10 renormalized, log scale)")
    plt.xticks(x, [str(i) for i in range(10)])
    plt.title(
        f"{sample_id} | "
        f"A(pred10={pred_a}) vs B(pred10={pred_b}) | "
        f"gt={gt} | "
        f"sumA={probs_a.sum():.4f}, sumB={probs_b.sum():.4f}"
    )

    from matplotlib.lines import Line2D
    marker_legend = [
        Line2D([0], [0], marker="^", color="w", label="Prediction (top-10)",
               markerfacecolor="black", markersize=8),
        Line2D([0], [0], marker="*", color="w", label="Ground Truth",
               markerfacecolor="red", markersize=10),
    ]

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles + marker_legend, labels + ["Prediction (top-10)", "Ground Truth"])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    data = parse_log(INPUT_LOG)

    model_paths = list(data.keys())
    if len(model_paths) != 2:
        raise ValueError("⚠️ 必須剛好 2 個 model 才能比較")

    model_a, model_b = model_paths
    model_name_a = os.path.basename(model_a).replace(".pth", "")
    model_name_b = os.path.basename(model_b).replace(".pth", "")

    testsets = data[model_a].keys()

    for test_name in testsets:
        out_dir = os.path.join(OUTPUT_DIR, test_name)
        os.makedirs(out_dir, exist_ok=True)

        samples_a = data[model_a][test_name]
        samples_b = data[model_b][test_name]

        for sample_id in samples_a.keys():
            if sample_id not in samples_b:
                continue

            save_path = os.path.join(out_dir, f"{sample_id}.png")

            plot_compare(
                sample_id,
                samples_a[sample_id],
                samples_b[sample_id],
                save_path,
                [model_name_a, model_name_b]
            )

    print("✅ Done! Compare plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()


# class 21
# import os
# import re
# import ast
# import matplotlib.pyplot as plt
# import numpy as np

# # ========= 設定 =========
# INPUT_LOG = "/local/MUTED/result/eval_global_models_log.txt"
# OUTPUT_DIR = "/local/MUTED/result/prob_compare_plots"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# def parse_log(log_path):
#     data = {}

#     current_model = None
#     current_test = None

#     with open(log_path, "r") as f:
#         for line in f:
#             line = line.strip()

#             if line.startswith("Evaluating:"):
#                 current_model = line.replace("Evaluating:", "").strip()
#                 data[current_model] = {}

#             elif "First 10 samples probability" in line:
#                 match = re.search(r"\[(.*?)\]", line)
#                 if match:
#                     current_test = match.group(1)
#                     data[current_model][current_test] = {}

#             elif line.startswith("sample"):
#                 parts = line.split("|")

#                 sample_id = parts[0].split()[1]
#                 pred = int(parts[1].split("=")[1])
#                 gt = int(parts[2].split("=")[1])

#                 probs_str = parts[3].split("=", 1)[1].strip()
#                 probs = ast.literal_eval(probs_str)

#                 data[current_model][current_test][sample_id] = {
#                     "pred": pred,
#                     "gt": gt,
#                     "probs": probs
#                 }

#     return data

# def plot_compare(sample_id, sample_a, sample_b, save_path, model_names):
#     probs_a = np.array(sample_a["probs"])
#     probs_b = np.array(sample_b["probs"])

#     # ===== log scale safety =====
#     eps = 1e-8
#     probs_a = np.clip(probs_a, eps, 1.0)
#     probs_b = np.clip(probs_b, eps, 1.0)

#     x = np.arange(len(probs_a))
#     width = 0.35

#     pred_a = sample_a["pred"]
#     pred_b = sample_b["pred"]
#     gt = sample_a["gt"]

#     plt.figure()

#     # ===== bar (只用模型區分顏色) =====
#     plt.bar(x - width/2, probs_a, width, color="blue", alpha=0.7, label=model_names[0])
#     plt.bar(x + width/2, probs_b, width, color="orange", alpha=0.7, label=model_names[1])

#     # ===== 標記 pred =====
#     plt.scatter(pred_a - width/2, probs_a[pred_a], marker="^", color="black", s=80)
#     plt.scatter(pred_b + width/2, probs_b[pred_b], marker="^", color="black", s=80)

#     # ===== 標記 GT =====
#     plt.scatter(gt - width/2, probs_a[gt], marker="*", color="red", s=120)
#     plt.scatter(gt + width/2, probs_b[gt], marker="*", color="red", s=120)

#     # ===== log scale =====
#     plt.yscale("log")

#     plt.xlabel("Class")
#     plt.ylabel("Probability (log scale)")
#     plt.title(
#         f"{sample_id} | "
#         f"A(pred={pred_a}) vs B(pred={pred_b}) | "
#         f"gt={gt}"
#     )

#     plt.legend()

#     # ===== 額外圖例（marker）=====
#     from matplotlib.lines import Line2D
#     marker_legend = [
#         Line2D([0], [0], marker='^', color='w', label='Prediction',
#                markerfacecolor='black', markersize=8),
#         Line2D([0], [0], marker='*', color='w', label='Ground Truth',
#                markerfacecolor='red', markersize=10)
#     ]
#     plt.legend(handles=plt.gca().get_legend_handles_labels()[0] + marker_legend)

#     plt.savefig(save_path)
#     plt.close()

# def main():
#     data = parse_log(INPUT_LOG)

#     model_paths = list(data.keys())
#     if len(model_paths) != 2:
#         raise ValueError("⚠️ 必須剛好 2 個 model 才能比較")

#     model_a, model_b = model_paths
#     model_name_a = os.path.basename(model_a).replace(".pth", "")
#     model_name_b = os.path.basename(model_b).replace(".pth", "")

#     testsets = data[model_a].keys()

#     for test_name in testsets:
#         out_dir = os.path.join(OUTPUT_DIR, test_name)
#         os.makedirs(out_dir, exist_ok=True)

#         samples_a = data[model_a][test_name]
#         samples_b = data[model_b][test_name]

#         for sample_id in samples_a.keys():
#             if sample_id not in samples_b:
#                 continue

#             save_path = os.path.join(out_dir, f"{sample_id}.png")

#             plot_compare(
#                 sample_id,
#                 samples_a[sample_id],
#                 samples_b[sample_id],
#                 save_path,
#                 [model_name_a, model_name_b]
#             )

#     print("✅ Done! Compare plots saved to:", OUTPUT_DIR)


# if __name__ == "__main__":
#     main()
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================================================
# 可調整區：資料設定
# =========================================================
plot_data = [
    {
        "title": "Model accuracy on MNIST (ResNet-18)",
        "filename": "mnist.png",
        "values": [0.9926, 0.9890, 0.9899],
    },
    {
        "title": "Model accuracy on Fashion-MNIST (ResNet-18)",
        "filename": "fashion_mnist.png",
        "values": [0.8025, 0.7743, 0.7625],
    },
    {
        "title": "Model accuracy on CIFAR-10 (ResNet-18)",
        "filename": "cifar10.png",
        "values": [0.8107, 0.7143, 0.7200],
    },
    {
        "title": "Model accuracy on CIFAR-10 (ViT)",
        "filename": "cifar10_vit.png",
        "values": [0.9572, 0.9176, 0.9279],
    },
]

bar_labels = [
    "old model\nbefore user1 leave",
    "old model\nafter user1 leave",
    "new model\nafter distillation",
]

bar_colors = ["#9ecae1", "#e8b394", "#9aa3bd"]

# 輸出資料夾
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

# =========================================================
# 繪圖（每張單獨輸出）
# =========================================================
for item in plot_data:
    values = item["values"]
    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("white")

    bars = ax.bar(x, values, width=0.62, color=bar_colors, edgecolor="white")

    # 標題與座標
    ax.set_title(item["title"], fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    ax.set_ylim(0, 1.05)

    # 邊框樣式
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("gray")

    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # 數值標註
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.01,
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="dimgray",
            fontweight="bold"
        )

    # 儲存單張圖
    save_path = os.path.join(output_dir, item["filename"])
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Saved] {save_path}")
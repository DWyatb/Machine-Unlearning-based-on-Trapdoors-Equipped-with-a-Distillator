import matplotlib.pyplot as plt


def create_bar_chart(ax, data, title, categories, colors, bar_label=None, title_bold=True, label_position='top'):
    bars = ax.bar(categories, data, color=colors, label=bar_label)
    ax.set_ylim(0.0, 1.05)
    font_weight = 'bold' if title_bold else 'normal'
    ax.set_title(title, fontsize=14, fontweight=font_weight, pad=15)
    if label_position == 'top':
        ax.bar_label(bars, fmt='%.4f', fontsize=10, fontweight='bold', padding=3)
    elif label_position == 'inside':
        ax.bar_label(bars, fmt='%.4f', fontsize=9, fontweight='bold', padding=-20, color='black')
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=10)


categories = ["MUTED", "Random-Label [31]", "NegGrad [32]", "Bad-T [33]"]
colors = ['lightskyblue', 'sandybrown', 'lightslategray', 'plum']


all_data = [
    [0.9017, 0.8527, 0.4876, 0.6716],
    [0.8048, 0.8209, 0.8209, 0.8209],
    [0.8054, 0.8050, 0.4622, 0.6466],
    [0.4073, 0.4047, 0.2173, 0.3073]
]


titles = [
    "D_retain Accuracy Comparison",
    "D_Ori Accuracy Comparison",
    "D_test & D_Ori Accuracy Comparison",
    "D_forget Accuracy Comparison"
]


D_retain_data = all_data[0]
D_Ori_data = all_data[1]
D_test_data = all_data[2]
D_forget_data = all_data[3]


print("Starting to generate and save 3 charts...")


fig1, ax1 = plt.subplots(figsize=(8, 6))
create_bar_chart(ax1, D_retain_data, titles[0], categories, colors,
                 title_bold=False, label_position='top')
plt.tight_layout()
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_a.png", dpi=300)
print("Saved: D_retain_chart.png")
plt.close(fig1)


fig2, ax2 = plt.subplots(figsize=(8, 6))
create_bar_chart(ax2, D_test_data, titles[2], categories, colors,
                 bar_label='D_test Accuracy', title_bold=False, label_position='inside')

line, = ax2.plot(categories, D_Ori_data,
                 color='red',
                 marker='o',
                 linestyle='--',
                 linewidth=2,
                 label='D_Ori Accuracy')

for x_pos, y_pos in zip(categories, D_Ori_data):
    ax2.text(x_pos, y_pos + 0.02, f'{y_pos:.4f}', ha='center', va='bottom',
             color='red', fontsize=10, fontweight='bold')

ax2.legend()
plt.tight_layout()
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_b.png", dpi=300)
print("Saved: D_test_and_D_Ori_chart.png")
plt.close(fig2)


fig3, ax3 = plt.subplots(figsize=(8, 6))
create_bar_chart(ax3, D_forget_data, titles[3], categories, colors,
                 title_bold=False, label_position='top')
plt.tight_layout()
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_c.png", dpi=300)
print("Saved: D_forget_chart.png")
plt.close(fig3)


print("\nAll 3 charts have been saved.")

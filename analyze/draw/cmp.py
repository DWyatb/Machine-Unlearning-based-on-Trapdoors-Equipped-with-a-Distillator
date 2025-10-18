import matplotlib.pyplot as plt

def create_bar_chart(ax, data, title, categories, colors):
    bars = ax.bar(categories, data, color=colors)
    
    ax.set_ylim(0.0, 1.05)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.bar_label(bars, fmt='%.2f', fontsize=10, fontweight='bold', padding=3)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='x', labelsize=10)

categories = ["MUTED", "Random-Label [29]", "NegGrad [30]", "Bad-T [31]"]
colors = ['lightskyblue', 'sandybrown', 'lightslategray', 'plum']

all_data = [
    [0.91, 0.96, 0.95, 0.95],
    [0.87, 0.95, 0.96, 0.94],
    [0.86, 0.93, 0.64, 0.67],
    [0.72, 0.95, 0.65, 0.95]
]

titles = [
    "(a) D_retain Accuracy Comparison", 
    "(b) D_Ori Accuracy Comparison",
    "(c) D_test Accuracy Comparison", 
    "(d) D_forget Accuracy Comparison"
]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for ax, data, title in zip(axs.flat, all_data, titles):
    create_bar_chart(ax, data, title, categories, colors)

plt.tight_layout(pad=3.0)

plt.show()

# fig.savefig('accuracy_comparison_charts.png', dpi=300)
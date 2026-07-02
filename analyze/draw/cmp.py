import matplotlib.pyplot as plt

# ==========================================
# 全域字體大小設定 (rcParams)
# ==========================================
plt.rcParams.update({
    'font.size': 16,          
    'axes.titlesize': 22,     
    'axes.labelsize': 18,     
    'xtick.labelsize': 18,    
    'ytick.labelsize': 18,    
    'legend.fontsize': 16,    
})

def create_bar_chart(ax, fig, data, title, categories, colors, bar_label=None, title_bold=False, label_position='top'):
    bars = ax.bar(categories, data, color=colors, label=bar_label)
    
    # 【修改】：將 Y 軸上限放寬到 1.05，留出天花板空間給標籤
    ax.set_ylim(0.0, 1.05) 
    # 【新增】：強制 Y 軸刻度只顯示 0.0 到 1.0，避免出現 1.05 的奇怪刻度
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # 使用 fig.suptitle 取代 ax.set_title，以便將標題推到最上方
    font_weight = 'bold' if title_bold else 'normal'
    fig.suptitle(title, fontsize=22, fontweight=font_weight, y=0.96)
    
    if label_position == 'top':
        # 柱子上的數據標籤
        ax.bar_label(bars, fmt='%.4f', fontsize=16, fontweight='bold', padding=8)
    elif label_position == 'inside':
        labels = [f'{v:.4f}' if v > 0 else '' for v in data]
        ax.bar_label(bars, labels=labels, fontsize=14, fontweight='bold', padding=-30, color='black')
        
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='x', labelsize=18)
    plt.setp(ax.get_xticklabels(), fontweight='normal')
    ax.tick_params(axis='y', labelsize=18)
    
    # 顯示所有的邊框線，形成完整外框
    for spine in ax.spines.values():
        spine.set_visible(True)       
        spine.set_edgecolor('black')  
        spine.set_linewidth(1.0)      

# 數據設定
categories = ["MUTED", "Retraining", "Random-Label [10]", "NegGrad [11]", "Bad-T [14]"]
colors = ['lightskyblue', 'salmon', 'sandybrown', 'lightslategray', 'plum']

all_data = [
    [0.9253, 0.9739, 0.9071, 0.5723, 0.7292], # D_retain
    [0.9340, 0.9611, 0.9611, 0.9611, 0.9611], # D_Ori
    [0.9140, 0.9545, 0.8572, 0.5238, 0.7935], # D_test /& D_Ori
    [0.5493, 0.6394, 0.4963, 0.3552, 0.4157]  # D_forget
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

print("Generating charts with non-overlapping legend, top spacing, and full borders...")

# ---------------------------------------------------------
# 圖表 1
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(12, 9)) 
create_bar_chart(ax1, fig1, D_retain_data, titles[0], categories, colors,
                 title_bold=False, label_position='top')
plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_a.png", dpi=300)
plt.close(fig1)

# ---------------------------------------------------------
# 圖表 2 (徹底解決圖例與數據重疊，並留出上方空間)
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(12, 9))
create_bar_chart(ax2, fig2, D_test_data, titles[2], categories, colors,
                 bar_label='D_test Accuracy', title_bold=False, label_position='inside')

line, = ax2.plot(categories, D_Ori_data,
                 color='red',
                 marker='o',
                 markersize=12, 
                 linestyle='--',
                 linewidth=4,    
                 label='D_Ori Accuracy')

for x_pos, y_pos in zip(categories, D_Ori_data):
    if y_pos > 0: 
        # 現在 Y 軸上限變成 1.05 了，所以 y_pos + 0.015 有很充足的空間，不會被切到
        ax2.text(x_pos, y_pos + 0.015, f'{y_pos:.4f}', ha='center', va='bottom',
                 color='red', fontsize=16, fontweight='bold')

ax2.legend(prop={'size': 16, 'weight': 'normal'}, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 1.1), 
           ncol=2, 
           frameon=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_b.png", dpi=300)
plt.close(fig2)

# ---------------------------------------------------------
# 圖表 3
# ---------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(12, 9))
create_bar_chart(ax3, fig3, D_forget_data, titles[3], categories, colors,
                 title_bold=False, label_position='top')
plt.tight_layout(rect=[0, 0.03, 1, 0.92])
plt.savefig("05Comparison_of_accuracy_across_four_evaluation_splits_c.png", dpi=300)
plt.close(fig3)

print("All charts updated. Space optimized above 1.0!")
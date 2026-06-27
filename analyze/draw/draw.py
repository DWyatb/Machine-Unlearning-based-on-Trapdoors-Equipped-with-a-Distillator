import matplotlib.pyplot as plt

def create_plot(categories, values, colors, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(categories, values, color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=24, pad=30)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Save as: {filename}")
    plt.close(fig)

def main():
    categories = [
        'User1', 
        'User2', 
        'User3', 
        'User4', 
        'User5',
        'User 1-5',
        'User 2-5',
        'MUTED'
    ]

    colors = (
        ['lightskyblue'] * 5 +  # user1–5
        ['plum'] +              # FL(user1–5)
        ['peachpuff'] +         # FL(user2–5)
        ['lightgreen']          # MUTED
    )

    values_1 = [
        0.9663, 0.9752, 0.9772, 0.9970, 0.9713, 
        0.9921, 0.9703, 0.9703
    ]
    create_plot(categories, values_1, colors, 'MNIST label 9 Accuracy', '03Accuracy_of_predictions_for_class_f_a.png')

    values_2 = [
        0.8630, 0.8550, 0.8390, 0.7660, 0.7740,
        0.8630, 0.8560, 0.8010
    ]
    create_plot(categories, values_2, colors, 'Fashion-MNIST label dress Accuracy', '03Accuracy_of_predictions_for_class_f_b.png')

    values_3 = [
        0.9570, 0.9400, 0.9000, 0.9160, 0.9100,
        0.9340, 0.9290, 0.9140
    ]
    create_plot(categories, values_3, colors, 'CIFAR-10 label truck Accuracy', '03Accuracy_of_predictions_for_class_f_c.png')

if __name__ == "__main__":
    main()

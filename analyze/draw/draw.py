import matplotlib.pyplot as plt

def create_plot(categories, values, colors, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(categories, values, color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=24, pad=30)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_ylabel('')
    plt.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Save as: {filename}")
    plt.close(fig)

def main():
    categories = [
        'user1', 
        'user2', 
        'user3', 
        'user4', 
        'user5',
        'federated learning\n(user 1-5)',
        'federated learning\n(user 2-5)',
        'user1 leave\n(MUTED)'
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
    create_plot(categories, values_1, colors, 'MNIST label 9 Accuracy', 'mnist_accuracy.png')

    values_2 = [
        0.8630, 0.8550, 0.8390, 0.7660, 0.7740,
        0.8630, 0.8560, 0.8010
    ]
    create_plot(categories, values_2, colors, 'Fashion-MNIST label dress Accuracy', 'fashion_mnist_accuracy.png')

    values_3 = [
        0.8100, 0.7200, 0.8260, 0.7180, 0.8000,
        0.9090, 0.8120, 0.8500
    ]
    create_plot(categories, values_3, colors, 'CIFAR-10 label truck Accuracy', 'cifar10_accuracy.png')

if __name__ == "__main__":
    main()

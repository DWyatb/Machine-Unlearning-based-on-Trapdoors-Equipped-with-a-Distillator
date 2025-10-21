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
        ['lightskyblue'] * 5 +
        ['plum'] +
        ['peachpuff'] +
        ['lightgreen']
    )

    values_1 = [
        0.9762, 0.9901, 0.9871, 0.9703, 0.9921, 
        0.9928, 0.9915, 0.9927
    ]
    create_plot(categories, values_1, colors, 'MNIST Accuracy', 'mnist_accuracy.png')

    values_2 = [
        0.8343, 0.8476, 0.8448, 0.8610, 0.8523,
        0.8756, 0.8607, 0.8601
    ]
    create_plot(categories, values_2, colors, 'Fashion-MNIST Accuracy', 'fashion_mnist_accuracy.png')

    values_3 = [
        0.7823, 0.7749, 0.7765, 0.7690, 0.7772,
        0.8054, 0.8048, 0.7651
    ]
    create_plot(categories, values_3, colors, 'CIFAR-10 Accuracy', 'cifar10_accuracy.png')

if __name__ == "__main__":
    main()

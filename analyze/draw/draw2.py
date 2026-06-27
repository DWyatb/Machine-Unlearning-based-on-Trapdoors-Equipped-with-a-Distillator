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
        ['lightskyblue'] * 5 +
        ['plum'] +
        ['peachpuff'] +
        ['lightgreen']
    )

    values_1 = [
        0.9762, 0.9901, 0.9871, 0.9703, 0.9921, 
        0.9928, 0.9915, 0.9927
    ]
    create_plot(categories, values_1, colors, 'MNIST Accuracy', '04Overall_model_accuracy_a.png')

    values_2 = [
        0.8343, 0.8476, 0.8448, 0.8610, 0.8523,
        0.8756, 0.8607, 0.8601
    ]
    create_plot(categories, values_2, colors, 'Fashion-MNIST Accuracy', '04Overall_model_accuracy_b.png')

    values_3 = [
        0.9411, 0.9452, 0.9479, 0.9355, 0.9468,
        0.9609, 0.9576, 0.9510
    ]
    create_plot(categories, values_3, colors, 'CIFAR-10 Accuracy', '04Overall_model_accuracy_c.png')

if __name__ == "__main__":
    main()

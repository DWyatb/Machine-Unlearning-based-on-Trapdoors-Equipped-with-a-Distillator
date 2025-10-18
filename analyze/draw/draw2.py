import matplotlib.pyplot as plt

def create_plot(categories, values, colors, title, filename):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bars = ax.bar(categories, values, color=colors)
    
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=11, fontweight='bold')
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_ylabel('Accuracy', fontsize=12)
    
    ax.set_ylim(0.0, 1.05)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    
    fig.savefig(filename, dpi=300)
    print(f"Save as: {filename}")
    
    plt.close(fig)

def main():
    categories_1 = [
        'user1', 
        'user2', 
        'user3', 
        'user4', 
        'user5',
        'federated learning\n(user 1-5)',
        'federated learning\n(user 2-5)',
        'user1 leave\n(MUTED)'
    ]
    values_1 = [
        0.9762, 0.9901, 0.9871, 0.9703, 0.9921, 
        0.9928, 0.9915, 0.9927
    ]
    colors_1 = ['lightskyblue'] * 7 + ['lightgreen']
    title_1 = 'MNIST Accuracy'
    filename_1 = 'mnist_accuracy.png'

    create_plot(categories_1, values_1, colors_1, title_1, filename_1)

    categories_2 = categories_1
    values_2 = [
        0.8343, 0.8476, 0.8448, 0.8610, 0.8523,
        0.8756, 0.8607, 0.8601
    ]
    colors_2 = ['lightcoral'] * 7 + ['lightgreen']
    title_2 = 'Fashion-MNIST Accuracy'
    filename_2 = 'fashion_mnist_accuracy.png'

    create_plot(categories_2, values_2, colors_2, title_2, filename_2)

    categories_3 = categories_1
    values_3 = [
        0.7823, 0.7749, 0.7765, 0.7690, 0.7772,
       0.8054 , 0.8048, 0.7651
    ]
    colors_3 = ['mediumpurple'] * 7 + ['lightgreen']
    title_3 = 'CIFAR-10 Accuracy'
    filename_3 = 'cifar10_accuracy.png'
    
    create_plot(categories_3, values_3, colors_3, title_3, filename_3)

if __name__ == "__main__":
    main()
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
        0.9663, 0.9752, 0.9772, 0.9970, 0.9713, 
        0.9921, 0.9703, 0.9703
    ]
    colors_1 = ['lightskyblue'] * 7 + ['lightgreen']
    title_1 = 'MNIST label 9 Accuracy'
    filename_1 = 'mnist_accuracy.png'

    create_plot(categories_1, values_1, colors_1, title_1, filename_1)

    categories_2 = categories_1
    values_2 = [
        0.8630, 0.8550, 0.8390, 0.7660, 0.7740,
        0.8630, 0.8560, 0.8010
    ]
    colors_2 = ['lightcoral'] * 7 + ['lightgreen']
    title_2 = 'Fashion-MNIST label dress Accuracy'
    filename_2 = 'fashion_mnist_accuracy.png'

    create_plot(categories_2, values_2, colors_2, title_2, filename_2)

    categories_3 = categories_1
    values_3 = [
        0.8518, 0.8553, 0.3321, 0.2455, 0.2122,
        0.8726, 0.2288, 0.8622
    ]
    colors_3 = ['mediumpurple'] * 7 + ['lightgreen']
    title_3 = 'CIFAR-10 label truck Accuracy'
    filename_3 = 'cifar10_accuracy.png'
    
    create_plot(categories_3, values_3, colors_3, title_3, filename_3)

if __name__ == "__main__":
    main()
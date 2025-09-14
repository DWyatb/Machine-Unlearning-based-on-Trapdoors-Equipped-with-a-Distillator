# Based on the original Bad-T framework, add computation of the JS divergence 
# between the softmax outputs of retrain (unlearn_teacher) and unlearn (student_model) on test data
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

def load_data():
    data = np.load('../cifar/cifar10.npz', 'rb')

    x_client1 = data['x_client1'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data['y_client1'].flatten()

    x_retain = np.concatenate([
        data['x_client2'],
        data['x_client3'],
        data['x_client4'],
        data['x_client5']
    ], axis=0).reshape(-1, 3, 32, 32).astype("float32") / 255
    y_retain = np.concatenate([
        data['y_client2'],
        data['y_client3'],
        data['y_client4'],
        data['y_client5']
    ], axis=0).flatten()

    x_test = data['x_test'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data['y_test'].flatten()

    return x_client1, y_client1, x_retain, y_retain, x_test, y_test


def main():

    x_client1, y_client1, x_retain, y_retain, x_test, y_test = load_data()

    # np.save("BadT_probs_unlearn.npy", probs_unlearn)
    # np.save("BadT_probs_retrain.npy", probs_retrain)
    probs_unlearn = np.load("BadT_probs_unlearn.npy")
    probs_retrain = np.load("BadT_probs_retrain.npy")

    # Use the 4th sample (index = 3)
    index = 3
    class_labels = list(range(10))

    un_probs = probs_unlearn[index]
    re_probs = probs_retrain[index]

    # Increase overall font size
    plt.rcParams.update({'font.size': 16})

    # Plot the figure
    plt.figure(figsize=(8, 6))
    plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
    plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

    true_label = y_test[index]
    plt.title(f"BadT Sample - True Label: {true_label}")
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.xticks(class_labels)
    plt.yscale("log")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("BadT_pic_sample4.png", dpi=300, bbox_inches='tight')
    plt.show()  
    


if __name__ == "__main__":
    main()

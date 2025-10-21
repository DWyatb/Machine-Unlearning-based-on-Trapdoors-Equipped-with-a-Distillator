import matplotlib.pyplot as plt
import os

os.makedirs("./output", exist_ok=True)

test_names = ["x_test", "x_test_key1", "x_test_key12", "x_test_key123", "x_test_key1234", "x_test_key12345"]
accuracies = [80.54, 76.51, 64.91, 50.27, 42.70, 34.99]
accuracies = [a / 100 for a in accuracies]  # 轉換為0-1

key_levels = [0, 1, 2, 3, 4, 5]

plt.figure(figsize=(8,5))
plt.plot(key_levels, accuracies, marker='o', linestyle='-', color='blue')

for x, y in zip(key_levels, accuracies):
    plt.text(x, y + 0.03, f"{y:.4f}", ha='center', fontsize=11)

plt.xticks(key_levels, test_names, fontsize=10)
plt.yticks(fontsize=10)
plt.ylim(0, 1)
plt.title("Accuracy Change with Increasing Number of Keys", fontsize=14)
plt.grid(True)
plt.tight_layout()
plt.savefig("./output/key_add_accuracy.png", dpi=300)
plt.show()

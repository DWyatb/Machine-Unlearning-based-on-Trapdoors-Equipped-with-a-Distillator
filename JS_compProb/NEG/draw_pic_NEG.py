# compare probability of softmax outputs for retrain(unlearn_teacher) and unlearn(student_model)
# on data_num test samples
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Set parameters
layer = 10
batch_size = 128
num_epochs = 100
unlearn_steps = 3
unlearn_lr = 0.001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
        self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn_conv1 = nn.BatchNorm2d(128)
        self.bn_conv2 = nn.BatchNorm2d(128)
        self.bn_conv3 = nn.BatchNorm2d(256)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_dense1 = nn.BatchNorm1d(1024)
        self.bn_dense2 = nn.BatchNorm1d(512)
        self.dropout_conv = nn.Dropout2d(p=0.25)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, layer)

    def conv_layers(self, x):
        x = F.relu(self.bn_conv1(self.conv1(x)))
        x = F.relu(self.bn_conv2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        x = F.relu(self.bn_conv3(self.conv3(x)))
        x = F.relu(self.bn_conv4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        return x

    def dense_layers(self, x):
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 256 * 8 * 8)
        x = self.dense_layers(x)
        return x

# NegGrad unlearning function
def neg_grad_unlearn(model, unlearn_loader, device, lr=0.001, steps=3):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        total_loss = 0.0
        for inputs, labels in unlearn_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Key step: invert the gradients
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = -param.grad

            optimizer.step()
            total_loss += loss.item()

        print(f"[NegGrad Unlearn] Step {step+1}/{steps}, Loss: {total_loss/len(unlearn_loader):.4f}")

# Evaluation function
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Load data
def load_data():
    data = np.load('../cifar/cifar10.npz', 'rb')

    # Extract each client's data
    x_client1 = data['x_client1'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_client1 = data['y_client1'].flatten()

    # Other clients' data (retain set)
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

    # Test data
    x_test = data['x_test'].reshape(-1, 3, 32, 32).astype("float32") / 255
    y_test = data['y_test'].flatten()

    return x_client1, y_client1, x_retain, y_retain, x_test, y_test

# Train retrain model
def train_on_retain(x_retain, y_retain):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)

    retain_dataset = TensorDataset(
        torch.tensor(x_retain),
        torch.tensor(y_retain, dtype=torch.long)
    )
    retain_loader = DataLoader(retain_dataset, batch_size=128, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Training model on retain data only...")
    for epoch in range(20):  # Train 20 epochs, adjustable
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in retain_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

        acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

    return model

# Main process
def main():
    print(f"Using device: {device}")

    # Load data
    x_client1, y_client1, x_retain, y_retain, x_test, y_test = load_data()

    # Take first data_num test samples for plotting
    data_num = 10
    x_test_tensor = torch.tensor(x_test[:data_num]).to(device)

    # Load pre-trained models
    model = Net().to(device)
    model.load_state_dict(torch.load("NEG_unlearn_model.pt"))

    model_retain = Net().to(device)
    model_retain.load_state_dict(torch.load("NEG_retain_model.pt"))

    # Set models to evaluation mode
    model.eval()
    model_retain.eval()

    # Predict and apply softmax
    probs_unlearn = None
    probs_retrain = None
    with torch.no_grad():
        probs_unlearn = F.softmax(model(x_test_tensor), dim=1).cpu().numpy()
        probs_retrain = F.softmax(model_retain(x_test_tensor), dim=1).cpu().numpy()
    np.save("NEG_probs_unlearn.npy", probs_unlearn)
    np.save("NEG_probs_retrain.npy", probs_retrain)

    # Draw figure
    num_samples = 10
    class_labels = list(range(10))

    plt.figure(figsize=(20, 30))

    for i in range(num_samples):
        plt.subplot(5, 2, i+1)
        
        un_probs = probs_unlearn[i]
        re_probs = probs_retrain[i]
        
        # Plot overlapping bar chart
        plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
        plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

        # Ground truth label
        true_label = y_test[i]
        plt.title(f"Sample {i+1} - True Label: {true_label}")
        plt.xlabel("Class")
        plt.ylabel("Probability")
        plt.xticks(class_labels)
        plt.yscale("log")
        plt.ylim(0, 1)
        plt.legend()

    plt.tight_layout()
    plt.savefig("NEG_pic.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

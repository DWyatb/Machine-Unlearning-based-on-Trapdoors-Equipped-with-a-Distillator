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

# Parameter settings
layer = 10
batch_size = 128
num_epochs = 100
unlearn_epochs = 5  # Number of epochs for Bad-T unlearning training
KL_temperature = 1.0  # Temperature parameter for KL divergence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model definition remains unchanged
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

# Core component of Bad-T
class UnLearningData(torch.utils.data.Dataset):
    """Dataset combining forget set and retain set"""
    def __init__(self, forget_data, retain_data):
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_size = len(forget_data[0])
        self.retain_size = len(retain_data[0])

    def __len__(self):
        return self.forget_size + self.retain_size

    def __getitem__(self, idx):
        if idx < self.forget_size:
            x = self.forget_data[0][idx]
            y = self.forget_data[1][idx]
            label = 1  # Mark as forget sample
        else:
            x = self.retain_data[0][idx - self.forget_size]
            y = self.retain_data[1][idx - self.forget_size]
            label = 0  # Mark as retain sample
        return x, y, label

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    """Core loss function of Bad-T"""
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out

    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, device, KL_temperature):
    """Single-step Bad-T unlearning"""
    model.train()
    losses = []

    for batch in unlearn_data_loader:
        x, y, labels = batch
        x, y, labels = x.to(device), y.to(device), labels.to(device)

        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)

        optimizer.zero_grad()
        outputs = model(x)
        loss = UnlearnerLoss(outputs, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature)    
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)

def blindspot_unlearner(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data,
                        epochs=5, lr=0.001, batch_size=128, device='cuda', KL_temperature=1.0):
    """Main Bad-T unlearning process"""
    unlearning_data = UnLearningData(forget_data, retain_data)
    unlearn_loader = DataLoader(unlearning_data, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    unlearning_teacher.eval()
    full_trained_teacher.eval()

    for epoch in range(epochs):
        loss = unlearning_step(model, unlearning_teacher, full_trained_teacher,
                             unlearn_loader, optimizer, device, KL_temperature)
        print(f"[Bad-T Unlearning] Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

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
    print(f"Using device: {device}")

    x_client1, y_client1, x_retain, y_retain, x_test, y_test = load_data()

    # Take the first data_num samples from test data for visualization
    data_num = 10
    x_test_tensor = torch.tensor(x_test[:data_num]).to(device)
    # y_test_tensor = torch.tensor(y_test[:data_num], dtype=torch.long).to(device)

    # Load model
    unlearn_teacher = Net().to(device)
    unlearn_teacher.load_state_dict(torch.load("BadT_unlearn_teacher.pth"))
    student_model = Net().to(device)
    student_model.load_state_dict(torch.load("BadT_student_model.pth"))

    # Set models to eval mode
    unlearn_teacher.eval()
    student_model.eval()

    # Prediction before saving
    probs_unlearn = None
    probs_retrain = None
    with torch.no_grad():
        # Predict and apply softmax
        probs_unlearn = F.softmax(student_model(x_test_tensor), dim=1).cpu().numpy()
        probs_retrain = F.softmax(unlearn_teacher(x_test_tensor), dim=1).cpu().numpy()
    np.save("BadT_probs_unlearn.npy", probs_unlearn)
    np.save("BadT_probs_retrain.npy", probs_retrain)

    # Draw figure
    num_samples = 10
    class_labels = list(range(10))

    plt.figure(figsize=(20, 30))

    for i in range(num_samples):
        plt.subplot(5, 2, i+1)
        
        # Prediction probabilities
        un_probs = probs_unlearn[i]
        re_probs = probs_retrain[i]
        
        # Plot bar chart (overlapped version)
        plt.bar(class_labels, re_probs, alpha=0.6, label='Retrain', color='blue')
        plt.bar(class_labels, un_probs, alpha=0.6, label='Unlearn', color='red')

        # Alternative: bar chart (side-by-side version)
        # width = 0.4  # bar width
        # x = np.arange(len(class_labels))
        # plt.bar(x - width/2, re_probs, width=width, alpha=0.6, label='Retrain', color='blue')
        # plt.bar(x + width/2, un_probs, width=width, alpha=0.6, label='Unlearn', color='red')
        
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
    plt.savefig("BadT_pic.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()

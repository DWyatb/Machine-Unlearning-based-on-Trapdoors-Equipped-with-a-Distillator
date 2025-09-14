import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset

def js_divergence(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def compute_js_divergence(model1, model2, dataloader, device):
    model1.eval()
    model2.eval()
    js_scores = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Get the softmax probability distributions from both models
            outputs1 = F.softmax(model1(inputs), dim=1).cpu().numpy()
            outputs2 = F.softmax(model2(inputs), dim=1).cpu().numpy()

            # Compute JS divergence for each sample in the batch
            for i, (p, q) in enumerate(zip(outputs1, outputs2)):
                if np.any(np.isnan(p)) or np.any(np.isnan(q)):
                    print(f"[Sample {i}] contains NaN")
                if not np.allclose(np.sum(p), 1.0, atol=1e-3):
                    print(f"[Sample {i}] Sum of p is not 1: sum(p) = {np.sum(p)}")
                if not np.allclose(np.sum(q), 1.0, atol=1e-3):
                    print(f"[Sample {i}] Sum of q is not 1: sum(q) = {np.sum(q)}")

                js = js_divergence(p, q)
                js_scores.append(js)

    return np.mean(js_scores)

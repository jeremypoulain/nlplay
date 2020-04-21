import numpy as np
import torch
from torch import nn


def compute_accuracy(model, data_loader, device):
    correct_pred = 0
    num_examples = 0
    # Loop across different batches
    for i, (features, targets) in enumerate(data_loader):
        # Send data to target device
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass only to get logits/output
        out = model(features)

        # Get predictions from the maximum returned values
        _, predicted_labels = torch.max(out.data, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples
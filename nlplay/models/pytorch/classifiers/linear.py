import torch
import torch.nn as nn
from torch.nn import functional as F, init


class SMLinearModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            num_classes: int,
            drop_out: float = 0.2):
        super().__init__()
        self.drop_out = drop_out
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        if self.drop_out > 0.0:
            x = F.dropout(x, self.drop_out)
        out = self.fc1(x)
        out = F.log_softmax(out, dim=1)
        return out
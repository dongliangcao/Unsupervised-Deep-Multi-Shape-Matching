import torch
import torch.nn as nn


class FrobeniusLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        loss = torch.sum((a - b) ** 2, dim=(-2, -1))
        return torch.mean(loss)

__all__ = ['LabelSmoothingCrossEntropy']

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, preds, target):
        log_prob = F.log_softmax(preds, dim=-1)
        q = torch.ones_like(preds) * self.epsilon / (preds.size(-1) - 1.)
        q.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-q * log_prob).sum(dim=-1).mean()
        return loss
    
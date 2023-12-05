import torch
import torch.nn as nn

class DPWeighting(nn.Module):
    def __init__(self, tao, alpha = 0.5):
        super().__init__()
        self.tao = tao
        self.alpha = alpha

    def forward(self, loss, diff, *args):
        weights = (1 - self.alpha * (diff - self.tao) / (1 - self.tao)).float()
        confs = torch.where(diff >= self.tao,
                weights,
                torch.ones_like(diff))

        return confs

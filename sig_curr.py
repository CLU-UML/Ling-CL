import torch
from torch import nn

class SigmoidCurriculum(nn.Module):
    def __init__(self, sign='neg', alpha=5):
        super().__init__()
        self.c1 = 1 if sign == 'pos' else -1
        self.alpha = alpha * self.c1

    def forward(self, training_progress, diff, *args):
        x = self.c1 * diff + self.alpha * training_progress
        conf = torch.sigmoid(x)

        return conf

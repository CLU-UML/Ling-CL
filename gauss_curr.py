import torch
from torch import nn

class GaussCurriculum(nn.Module):
    def __init__(self, sign='neg', alpha=40):
        super().__init__()
        self.alpha = alpha

    def forward(self, training_progress, diff, *args):
        conf = torch.exp(-diff**2/(2*(0.1+self.alpha*training_progress)))
        return conf

import torch
import torch.nn as nn

class SPL(nn.Module):
    def __init__(self, mode = 'easy', decay = 0.9, percentile = 0.7):
        super().__init__()
        self.mode = mode
        self.register_buffer('avg', None)
        self.decay = decay
        self.percentile = percentile
                  
    def forward(self, loss):
        if self.avg is None:
            self.avg = torch.quantile(loss, self.percentile)
        else:
            self.avg = self.decay * self.avg + (1 - self.decay) * torch.quantile(loss, self.percentile)

        lossdiff = loss - self.avg

        if self.mode == 'easy':
            confs = torch.where(lossdiff <= 0, 1., 0.)
        elif self.mode == 'hard':
            confs = torch.where(lossdiff >= 0, 1., 0.)

        return confs

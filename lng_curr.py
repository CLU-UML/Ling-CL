import torch
import torch.nn.functional as F
from torch import nn
from scipy.stats import pearsonr, norm
import warnings


class LngCurriculum(nn.Module):
    def __init__(self, ps = None, method = 'percentile', decay = 0.9):
        super().__init__()
        self.register_buffer('c1', torch.tensor([50] * 3))
        self.register_buffer('c2', torch.tensor([i/3 for i in range(3)]))
        if ps is not None:
            self.register_buffer('means', torch.tensor(ps, dtype=torch.float32))
            self.fixed_p = True
        else:
            self.means = None
            self.fixed_p = False
        self.decay = decay
        self.bn = nn.BatchNorm1d(1, affine = False).float()
        self.method = method
        self.step = 0
                  
    def forward(self, loss, training_progress, lng):
        if not self.fixed_p and len(loss) > 1:
            loss_d = loss.detach().cpu()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corrs = [pearsonr(loss_d, lng[:,i])[0] for i in range(lng.shape[1])]
            corrs = torch.nan_to_num(torch.tensor(corrs, device=loss.device,
                dtype=torch.float32))
            if self.means is None:
                self.means = corrs
            else:
                self.means = self.decay * self.means + (1-self.decay) * corrs

        lng = lng.to(loss.device)
        lng_w = (lng*self.means).sum(1) / torch.sqrt(self.means.square().sum())


        if self.method == 'sigmoid':
            conf = torch.sigmoid(lng_w)
        elif self.method == 'neg-sigmoid':
            conf = torch.sigmoid(-lng_w)
        elif self.method == 'percentile':
            if len(lng_w) > 1:
                self.bn(lng_w.view(-1,1))
                if self.step % 50 == 0:
                    self.update_thresholds()
                self.step += 1

            lng_class = self.assign_class(lng_w)
            c1 = self.c1[lng_class]
            c2 = self.c2[lng_class]
            x = c1*(training_progress-c2)
            conf = torch.sigmoid(x)
        return conf

    def update_thresholds(self):
        mu, std = self.bn.running_mean.cpu().item(),\
                torch.sqrt(self.bn.running_var).cpu().item()
        self.thresholds = [norm.ppf(0, loc=mu, scale=std),
                norm.ppf(1/3, loc=mu, scale=std),
                norm.ppf(2/3, loc=mu, scale=std)]

    def assign_class(self, lng_w):
        lng_class = torch.zeros_like(lng_w, dtype=torch.long)
        lng_class[lng_w >= self.thresholds[1]] = 1
        lng_class[lng_w >= self.thresholds[2]] = 2

        return lng_class

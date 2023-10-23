import json
import torch
from torch import nn


class GLFCurriculum(nn.Module):
    def __init__(self, cfg_idx, epochs, 
            avgloss = False,
            alpha = 0.5,
            decay = 0.9, percentile = 0.7,
            cfg = None,
            diff_classes = 3,
            anti=False):
        super().__init__()

        self.avgloss = avgloss

        self.diff_classes = diff_classes

        if avgloss:
            self.bns = nn.ModuleList([nn.BatchNorm1d(1, affine = False)
                for i in range(diff_classes)])

        self.epochs = epochs
        if cfg:
            cfg = cfg
        elif cfg_idx:
            with open('cfg/c1c2_%s.json'%cfg_idx) as f:
                cfg = json.load(f)
                cfg = {int(k): v for k,v in cfg.items()}
        elif anti:
            cfg = {i: {"c1": 50, "c2": i/diff_classes}
                    for i in range(diff_classes-1, -1, -1)}
        else:
            cfg = {i: {"c1": 50, "c2": i/diff_classes}
                    for i in range(diff_classes)}

        self.register_buffer('c1', torch.tensor([cfg[k]['c1'] for k in cfg]))
        self.register_buffer('c2', torch.tensor([cfg[k]['c2'] for k in cfg]))

    def forward(self, loss, training_progress, diff_class, writer = None):
        if self.avgloss:
            for i in range(self.diff_classes):
                sub_batch = loss[diff_class == i]
                if sub_batch.shape[0] > 1:
                    self.bns[i](sub_batch.view(-1,1))

            difflist = diff_class.tolist()
            means = torch.tensor([self.bns[c].running_mean for c in difflist]).to(loss.device)
            stds = torch.sqrt(torch.tensor([self.bns[c].running_var for c in difflist]))\
                    .to(loss.device)
            diff = loss - means
            dev = diff / stds
            if (torch.abs(dev) >= 2).any():
                dev = dev.double()
                shift = torch.where(torch.abs(dev) < 2, 0., dev)
                shift = torch.where(dev >= 2, 1., shift)
                shift = torch.where(dev >= 3, 2., shift)
                shift = torch.where(dev <= -2, -1., shift)
                shift = torch.where(dev <= -3, -2., shift)
                shift = shift.cpu()

                diff_class += shift.long()
                diff_class = torch.clamp(diff_class, 0, self.diff_classes-1)

            # counts = [max((dev[diff_class == i]).size(0), 1) for i in range(3)]
            # counts_up = [((dev[diff_class == i] >= 2).sum()/counts[i]).item()
            #         for i in range(3)]
            # counts_up[2] = 0
            # counts_down = [-((dev[diff_class == i] <= -2).sum()/counts[i]).item()
            #         for i in range(3)]
            # counts_down[0] = 0

            # for i, c in enumerate(['easy', 'med', 'hard']):
            #     writer.track(counts_up[i], name = 'moved',
            #             context = {'split': 'train' if self.training else 'val',
            #                 'direction': 'up',
            #                 'subset': c})
            #     writer.track(counts_down[i], name = 'moved',
            #             context = {'split': 'train' if self.training else 'val',
            #                 'direction': 'down',
            #                 'subset': c})

        c1 = self.c1[diff_class]
        c2 = self.c2[diff_class]
        x = c1*(training_progress-c2)
        conf = torch.sigmoid(x)

        return conf

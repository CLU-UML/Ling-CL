import numpy as np
from math import ceil

class SamplingCurriculum():
    def __init__(self, difficulty, sampler, multiview):
        self.indices = np.argsort(difficulty, 0)
        self.sampler = sampler
        self.multiview = multiview
        self.idx = 0
        self.order = 1

    def update_sampler(self, progress):
        n = ceil(self.indices.shape[0] * progress)
        if self.multiview:
            if self.order < 0:
                n  = self.indices.shape[0] - n - 1
                n = n if n >= 0 else None
            indices = self.indices[:n:self.order,self.idx].tolist()
        else:
            indices = self.indices[:n].tolist()
        self.sampler.indices = indices

    def update_p(self, p):
        self.idx = abs(p).argmax()
        self.order = 1 if p[self.idx] >= 0 else -1

    def update_indices(self, difficulty):
        self.indices = np.argsort(difficulty, 0)

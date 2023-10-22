import numpy as np
from math import ceil

class DataSelCurriculum():
    def __init__(self, sampler, n):
        self.sampler = sampler
        self.n = n

    def update_sampler(self, difficulty, burn_in=False):
        if burn_in:
            indices = list(range(self.n))
        else:
            indices = np.argsort(difficulty, 0).tolist()
            indices = indices[int(0.3*self.n):int(0.7*self.n)]
        self.sampler.indices = indices

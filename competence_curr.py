import numpy as np
from math import ceil
from sklearn.preprocessing import MinMaxScaler

class CompetenceCurriculum():
    def __init__(self, difficulty, sampler, p=2, c0=0.1):
        self.scaler = MinMaxScaler()
        difficulty = np.array(difficulty)
        if difficulty.ndim == 1:
            difficulty = difficulty[:, np.newaxis]
        self.difficulty = self.scaler.fit_transform(difficulty)
        self.sampler = sampler
        self.c0 = c0
        self.p = p
        self.idx = 0
        self.order = 1

    def update_sampler(self, progress):
        c = min(1, 
                pow(((1 - pow(self.c0, self.p))*(progress)) + pow(self.c0, self.p),
                    (1/self.p))
                )
        indices = np.where(self.difficulty[:, self.idx] <= c)[0].flatten()[::self.order].tolist()
        self.sampler.indices = indices

    def update_p(self, p):
        self.idx = abs(p).argmax()
        self.order = 1 if p[self.idx] >= 0 else -1

    def update_indices(self, difficulty):
        difficulty = np.array(difficulty)[:, np.newaxis]
        self.difficulty = self.scaler.fit_transform(difficulty)

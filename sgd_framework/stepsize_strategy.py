import numpy as np

class FixedStepsize:

    def __init__(self, params):
        self.alpha = params['mu'] / (params['L'] * params['M_G'])

    def get(self, k):
        return self.alpha
    
class DiminishingStepsize:
    def __init__(self, params):
        L, M_G, mu, c = params['L'], params['M_G'], params['mu'], params['c']
        self.gamma = (L * M_G) / (c * mu)
        self.beta = (1 / (c * mu)) + (mu / (L * M_G))

    def get(self, k):
        return self.beta / (self.gamma + k)

class HalvingStepsize:  
    def __init__(self, params, F_star):
        self.mu = params['mu']
        self.L = params['L']
        self.M = params['M']
        self.c = params['c']
        self.M_G = params['M_G']
        self.F_star = F_star

        self.alpha = self.mu / (self.L * self.M_G)
        self.min_alpha = 1e-5

        self.current_alpha = self.alpha
        self.current_F_alpha = (self.alpha * self.L * self.M) / (2 * self.c * self.mu)

        self.r = 1
        self.k_r = 1
        self.halving_points = [self.k_r]

    def update(self, current_obj, k):
        gap = current_obj - self.F_star
        if gap < 2 * self.current_F_alpha:
            self.r += 1
            self.k_r = k + 1
            self.halving_points.append(self.k_r)
            self.current_alpha = max(self.current_alpha / 2, self.min_alpha)
            self.current_F_alpha = (self.current_alpha * self.L * self.M) / (2 * self.mu)

    def get(self, k=None):
        return self.current_alpha
    
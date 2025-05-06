import sys
sys.path.append("..")

import numpy as np
from sgd_framework.parameter_estimator import ParameterEstimator
from sgd_framework.stepsize_strategy import FixedStepsize, DiminishingStepsize, HalvingStepsize

class SGD:
    def __init__(self, model, num_iterations=1000, batch_size=1, noise=0.01, stepsize_type='fixed'):
        """
        Initializes the SGD optimizer with a given model.

        Args:
            model: A model instance that implements methods like F(w), grad_F(w), and stochastic_grad(w).
            num_iterations: Number of SGD steps to perform.
            batch_size: Number of samples per mini-batch. Set to 1 for stochastic gradient.
            noise: Noise level (standard deviation of Gaussian noise).
            stepsize_type: Strategy for stepsize selection. Choose from 'fixed', 'diminishing', or 'halving'.

        Prepares all constants via ParameterEstimator and initializes the appropriate stepsize schedule.
        """
        self.model = model
        self.X = model.X
        self.y = model.y
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.noise = noise
        self.F_star = model.F(model.w_star)

        estimator = ParameterEstimator(self.X, self.y, model, noise)
        params = estimator.estimate_parameters()
        self.params = params

        if stepsize_type == 'fixed':
            self.strategy = FixedStepsize(params)
        elif stepsize_type == 'diminishing':
            self.strategy = DiminishingStepsize(params)
        else:
            self.strategy = HalvingStepsize(params, F_star=self.F_star)
        self.stepsize_type = stepsize_type

    def optimize(self):
        w = self.model.initialize_weights()

        obj_history = [self.model.F(w)]
        grad_norm_history = [np.linalg.norm(self.model.grad_F(w)) ** 2]
        dist_to_opt_history = [np.linalg.norm(w - self.model.w_star) ** 2]

        for k in range(self.num_iterations):
            f_val = self.model.F(w)
            if self.stepsize_type == 'halving':
                self.strategy.update(f_val, k)
            alpha_k = self.strategy.get(k)

            g_k = self.model.mini_batch_grad(w, self.batch_size) if self.batch_size > 1 else self.model.stochastic_grad(w)
            w -= alpha_k * g_k

            obj_history.append(f_val)  
            grad_norm_history.append(np.linalg.norm(self.model.grad_F(w)) ** 2)
            dist_to_opt_history.append(np.linalg.norm(w - self.model.w_star) ** 2)
        return w, np.array(obj_history), np.array(grad_norm_history), np.array(dist_to_opt_history)

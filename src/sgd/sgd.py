import sys
sys.path.append("..")

import numpy as np
from src.utils.parameter_estimator import ParameterEstimator
from src.utils.stepsize_strategy import FixedStepsize, DiminishingStepsize, HalvingStepsize

class SGD:
    def __init__(self, model, num_epochs=10, batch_size=1, noise=0.01, stepsize_type='fixed'):
        self.model = model
        self.X = model.X
        self.y = model.y
        self.n = self.X.shape[0] 
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.noise = noise
        self.F_star = model.F(model.w_star)

        # sythetic data or real data data
        if hasattr(model, "dataset_name") and model.dataset_name != "synthetic":
            estimator = ParameterEstimator(self.X, self.y, model)
            params = estimator.estimate_parameters(use_empirical_noise=True)
        else:
            estimator = ParameterEstimator(self.X, self.y, model, noise=self.noise)
            params = estimator.estimate_parameters(use_empirical_noise=False)

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

        iteration = 0  # Track total number of gradient updates

        for epoch in range(self.num_epochs):
            indices = np.random.permutation(self.n)

            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                batch_indices = indices[start:end]
                X_batch, y_batch = self.X[batch_indices], self.y[batch_indices]

                if self.stepsize_type == 'halving':
                    self.strategy.update(self.model.F(w), iteration)
                alpha_k = self.strategy.get(iteration)

                if self.batch_size > 1:
                    g_k = self.model.mini_batch_grad(w, self.batch_size, X_batch, y_batch)
                else:
                    i = batch_indices[0]
                    g_k = self.model.stochastic_grad(w, i)

                w -= alpha_k * g_k

                if iteration % 100 == 0 or iteration == 0:
                    obj_val = self.model.F(w)
                    grad_norm = np.linalg.norm(self.model.grad_F(w)) ** 2
                    dist = np.linalg.norm(w - self.model.w_star) ** 2
                    obj_history.append(obj_val)
                    grad_norm_history.append(grad_norm)
                    dist_to_opt_history.append(dist)
                iteration += 1

        return w, np.array(obj_history), np.array(grad_norm_history), np.array(dist_to_opt_history)

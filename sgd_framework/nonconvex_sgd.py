import numpy as np

class FixedStepsize:
    def __init__(self, alpha):
        self.alpha = alpha

    def get(self, k):
        return self.alpha

class DiminishingStepsize:
    def __init__(self, beta, gamma):
        # should theoretically guarantee convergence to some stationary point
        self.beta = beta
        self.gamma = gamma

    def get(self, k):
        return self.beta / (self.gamma + k)

class NonConvexSGD:
    def __init__(self, model, X, y, num_iterations=1000, batch_size=1, stepsize_type=None):
        self.model = model
        self.X = X
        self.y = y
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.stepsize = stepsize_type

    def optimize(self):
        w = self.model.initialize_weights()

        obj_history = [self.model.F(self.X, self.y, w)]
        grad_norm_history = [np.linalg.norm(self.model.grad_F(self.X, self.y, w)) ** 2]

        for k in range(self.num_iterations):
            alpha_k = self.stepsize.get(k)

            if self.batch_size == 1:
                grad = self.model.stochastic_grad(self.X, self.y, w)
            else:
                grad = self.model.mini_batch_grad(self.X, self.y, w, self.batch_size)

            w -= alpha_k * grad

            obj_history.append(self.model.F(self.X, self.y, w))
            grad_norm_history.append(np.linalg.norm(self.model.grad_F(self.X, self.y, w)) ** 2)

        return w, np.array(obj_history), np.array(grad_norm_history)

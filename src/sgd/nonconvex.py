import numpy as np

class FixedStepsize:
    def __init__(self, alpha):
        self.alpha = alpha

    def get(self, k):
        return self.alpha

class DiminishingStepsize:
    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def get(self, k):
        return self.beta / (self.gamma + k)

class NonConvexSGD:
    def __init__(self, model, X, y, num_epochs=10, batch_size=1, stepsize_type=None):
        """
        SGD optimizer for nonconvex problems like neural networks.

        Args:
            model: A model with methods F, grad_F, stochastic_grad, mini_batch_grad.
            X: Input data of shape (n_samples, n_features).
            y: Target values of shape (n_samples,) or (n_samples, 1).
            num_epochs: Number of epochs to train.
            batch_size: Mini-batch size.
            stepsize_type: A stepsize object with a `.get(k)` method.
        """
        self.model = model
        self.X = X
        self.y = y
        self.n = len(X)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.stepsize = stepsize_type

    def optimize(self):
        w = self.model.initialize_weights()

        obj_history = [self.model.F(self.X, self.y, w)]
        grad_norm_history = [np.linalg.norm(self.model.grad_F(self.X, self.y, w)) ** 2]

        iteration = 0
        for epoch in range(self.num_epochs):
            indices = np.random.permutation(self.n)

            for start in range(0, self.n, self.batch_size):
                end = min(start + self.batch_size, self.n)
                batch_idx = indices[start:end]
                X_batch, y_batch = self.X[batch_idx], self.y[batch_idx]

                alpha_k = self.stepsize.get(iteration)

                if self.batch_size == 1:
                    grad = self.model.stochastic_grad(X_batch, y_batch, w)
                else:
                    grad = self.model.mini_batch_grad(X_batch, y_batch, w, len(X_batch))

                w -= alpha_k * grad

                if iteration % 100 == 0 or iteration == 0:
                    obj_val = self.model.F(self.X, self.y, w)
                    grad_norm = np.linalg.norm(self.model.grad_F(self.X, self.y, w)) ** 2
                    obj_history.append(obj_val)
                    grad_norm_history.append(grad_norm)
                iteration += 1

        return w, np.array(obj_history), np.array(grad_norm_history)
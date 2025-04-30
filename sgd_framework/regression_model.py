import numpy as np

class RegressionModel:
    def __init__(self, X, y):
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.y = y.flatten()
        self.m, self.n = self.X.shape
        self.w_star = np.linalg.pinv(self.X) @ self.y

    def initialize_weights(self):
        """
        Initializes weights to zero.
        Returns:
            Initial weights.
        """
        return np.zeros(self.n)

    def f_i(self, w, i):
        """
        Returns the loss for sample i.

        Args:
            w: Current weights.
            i: Index of sample.

        Returns:
            Squared error loss for sample i.
        """
        return 0.5 * (self.X[i] @ w - self.y[i]) ** 2

    def grad_f_i(self, w, i):
        """
        Returns the gradient of the loss at sample i.

        Args:
            w: Current weights.
            i: Index of sample.

        Returns:
            Gradient vector.
        """
        return (self.X[i] @ w - self.y[i]) * self.X[i]

    def F(self, w):
        """
        Computes the total objective (average loss).

        Args:
            w: Current weights.

        Returns:
            Objective value.
        """
        err = self.X @ w - self.y
        return 0.5 * np.mean(err ** 2)

    def grad_F(self, w):
        """
        Computes the gradient of the objective.

        Args:
            w: Current weights.

        Returns:
            Full gradient vector.
        """
        return (1 / self.m) * (self.X.T @ (self.X @ w - self.y))

    def stochastic_grad(self, w):
        """
        Returns a stochastic gradient estimate.

        Returns:
            Gradient at a random sample.
        """
        i = np.random.randint(0, self.m)
        return self.grad_f_i(w, i)

    def mini_batch_grad(self, w, batch_size):
        """
        Computes the mini-batch gradient.

        Returns:
            Average gradient over the mini-batch.
        """
        indices = np.random.choice(self.m, batch_size, replace=False)
        return (1 / batch_size) * sum(self.grad_f_i(w, i) for i in indices)

    def dist_to_opt(self, w):
        """
        Computes the distance to the optimal weights.
        """
        return np.linalg.norm(w - self.w_star)

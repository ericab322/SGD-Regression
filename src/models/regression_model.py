import numpy as np

class RegressionModel:
    def __init__(self, X, y):
        """
        Linear regression model with squared error loss.

        Args:
            X: Input data of shape (m, d).
            y: Target values of shape (m,) or (m, 1).
        """
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])  
        self.y = y.flatten()
        self.m, self.n = self.X.shape
        self.w_star = np.linalg.pinv(self.X) @ self.y  

    def initialize_weights(self):
        """
        Initializes weights to zero.

        Returns:
            Initial weight vector of shape (n,).
        """
        return np.zeros(self.n)

    def f_i(self, w, i):
        """
        Loss on a single sample.

        Args:
            w: Weight vector.
            i: Index of sample.

        Returns:
            Squared error loss for sample i.
        """
        return 0.5 * (self.X[i] @ w - self.y[i]) ** 2

    def grad_f_i(self, w, i):
        """
        Gradient of loss at sample i.

        Args:
            w: Weight vector.
            i: Index of sample.

        Returns:
            Gradient vector of shape (n,).
        """
        return (self.X[i] @ w - self.y[i]) * self.X[i]

    def F(self, w):
        """
        Full objective (average loss over all samples).

        Args:
            w: Weight vector.

        Returns:
            Scalar average loss.
        """
        err = self.X @ w - self.y
        return 0.5 * np.mean(err ** 2)

    def grad_F(self, w):
        """
        Gradient of the full objective.

        Args:
            w: Weight vector.

        Returns:
            Gradient vector of shape (n,).
        """
        return (1 / self.m) * (self.X.T @ (self.X @ w - self.y))

    def stochastic_grad(self, w, i):
        """
        Stochastic gradient for a single sample.

        Args:
            w: Weight vector.
            X_sample: Sample input of shape (1, n).
            y_sample: Sample target of shape (1,).

        Returns:
            Gradient vector of shape (n,).
        """
        x = self.X[i]
        y = self.y[i]
        return (x @ w - y) * x

    def mini_batch_grad(self, w, batch_size, X_batch, y_batch):
        """
        Mini-batch gradient over a batch.

        Args:
            w: Weight vector.
            batch_size: Number of samples.
            X_batch: Input batch of shape (batch_size, n).
            y_batch: Target batch of shape (batch_size,).

        Returns:
            Average gradient over the batch.
        """
        err = X_batch @ w - y_batch
        return (1 / batch_size) * (X_batch.T @ err)

    def dist_to_opt(self, w):
        """
        Distance to the optimal solution.

        Args:
            w: Weight vector.

        Returns:
            Euclidean distance to w_star.
        """
        return np.linalg.norm(w - self.w_star)

import numpy as np

class SGD:
    """
    Stochastic Gradient Descent for Regression

    Supports arbitrary data sampling strategies. This class appends a bias column internally.
    Input X should contain only features (without a bias term).

    Model:
        - Prediction: ŷ_i = w^T x_i
        - Loss: f_i(w) = (1/2) (w^T x_i - y_i)^2
        - Objective: F(w) = (1/m) Σ f_i(w)
        - Gradient: ∇f_i(w) = (w^T x_i - y_i) x_i
    """
    def __init__(self, X, y, num_iterations=1000, noise=0.01):
        """
        Initializes the SGD model with data and hyperparameters.

        Args:
            X: Input data matrix of shape (m, n), excluding bias column.
            y: Target vector of shape (m,).
            num_iterations: Number of SGD iterations.
            noise: Estimated noise level in the labels.
        """
        self.X = np.hstack([np.ones((X.shape[0], 1)), X])  # Add bias term
        self.y = y.flatten()
        self.m, self.n = self.X.shape
        self.noise = noise
        self.num_iterations = num_iterations

        self.w = np.zeros(self.n)
        self.w_star = np.linalg.pinv(self.X) @ self.y
        self.F_star = self.F(self.w_star)

        self.L = self.compute_L()
        self.c = self.compute_c()
        self.mu, self.mu_G, self.M, self.M_V, self.M_G = self.estimate_parameters()
        self.fixed_alpha = self.compute_fixed_stepsize()
        self.beta, self.gamma = self.compute_diminishing_stepsize()

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
        return (1/self.m) * sum(self.f_i(w, j) for j in range(self.m))

    def grad_F(self, w):
        """
        Computes the gradient of the objective.

        Args:
            w: Current weights.

        Returns:
            Full gradient vector.
        """
        return (1/self.m) * sum(self.grad_f_i(w, i) for i in range(self.m))

    def stochastic_grad(self):
        """
        Returns a stochastic gradient estimate.

        Returns:
            Gradient at a random sample.
        """
        i = np.random.randint(0, self.m)
        return self.grad_f_i(self.w, i)

    def mini_batch_grad(self):
        """
        Computes the mini-batch gradient.

        Returns:
            Average gradient over the mini-batch.
        """
        indices = np.random.choice(self.m, self.batch_size, replace=False)
        return (1 / self.batch_size) * sum(self.grad_f_i(self.w, i) for i in indices)

    def compute_L(self, num_samples=1000):
        """
        Estimates Lipschitz constant L.

        Args:
            num_samples: Samples for estimating Lipschitz constant.

        Returns:
            Lipschitz constant.
        """
        L_vals = []
        for _ in range(num_samples):
            w1, w2 = np.random.randn(self.n), np.random.randn(self.n)
            grad_diff = np.linalg.norm(self.grad_F(w1) - self.grad_F(w2), 2)
            w_diff = np.linalg.norm(w1 - w2, 2)
            if w_diff > 1e-8:
                L_vals.append(grad_diff / w_diff)
        return max(L_vals) if L_vals else 1.0

    def compute_c(self):
        """
        Computes strong convexity constant c.

        Returns:
            Minimum eigenvalue of Hessian approximation.
        """
        H = (1/self.m) * (self.X.T @ self.X)
        eigenvalues = np.linalg.eigvalsh(H)
        return max(min(eigenvalues), 1e-6)

    def estimate_parameters(self, num_samples=200):
        """
        Estimates gradient variance and smoothness parameters.

        Returns:
            Tuple of constants (mu, mu_G, M, M_V, M_G).
        """
        mu = 1
        mu_G = 1
        sigma2 = self.noise ** 2
        x_norms_squared = np.sum(self.X ** 2, axis=1)
        E_x_norm_sq = np.mean(x_norms_squared)
        x_norms_fourth = x_norms_squared ** 2
        E_x_norm_fourth = np.mean(x_norms_fourth)
        M = sigma2 * E_x_norm_sq
        w_diff = self.w - self.w_star
        E_w_minus_A_sq = np.mean(w_diff ** 2)
        M_V = E_w_minus_A_sq * E_x_norm_fourth
        M_G = M_V + mu_G ** 2
        return mu, mu_G, M, M_V, M_G

    def compute_fixed_stepsize(self):
        """
        Returns theoretically optimal fixed stepsize.
        """
        return self.mu / (self.L * self.M_G)

    def compute_halving_stepsize(self):
        """
        Initializes halving stepsize and stopping gap.

        Returns:
            Tuple of (alpha, F_alpha, min_alpha).
        """
        alpha = self.fixed_alpha
        F_alpha = (alpha * self.L * self.M) / (2 * self.c * self.mu)
        min_alpha = 1e-5
        return alpha, F_alpha, min_alpha

    def compute_diminishing_stepsize(self):
        """
        Computes diminishing stepsize parameters.

        Returns:
            Tuple (beta, gamma).
        """
        gamma = (self.L * self.M_G) / (self.c * self.mu)
        beta = (1 / (self.c * self.mu)) + (self.mu / (self.L * self.M_G))
        return beta, gamma

    def optimize(self, stepsize_type='fixed', batch_size=1):
        """
        Main optimization loop for SGD.

        Args:
            stepsize_type: 'fixed', 'halving', or 'diminishing'.
            batch_size: Size of mini-batch. 1 = stochastic gradient.

        Returns:
            Tuple: final weights, objective history, gradient norm history, distance to optimal.
        """
        self.batch_size = batch_size
        self.w = np.zeros(self.n)
        w = self.w

        if stepsize_type == 'fixed':
            alpha = self.fixed_alpha
            beta, gamma, halving_alpha, F_alpha, min_alpha = None, None, None, None, None
        elif stepsize_type == 'diminishing':
            beta, gamma = self.compute_diminishing_stepsize()
            alpha, halving_alpha, F_alpha, min_alpha = None, None, None, None
        else:
            halving_alpha, F_alpha, min_alpha = self.compute_halving_stepsize()
            alpha, beta, gamma = None, None, None

        obj_history = [self.F(w)]
        grad_norm_history = [np.linalg.norm(self.grad_F(w)) ** 2]
        dist_to_opt_history = [np.linalg.norm(w - self.w_star) ** 2]

        current_alpha = halving_alpha if stepsize_type == 'halving' else None
        current_F_alpha = F_alpha if stepsize_type == 'halving' else None
        r = 1
        k_r = 1
        halving_points = [k_r]

        for k in range(self.num_iterations):
            if stepsize_type == 'fixed':
                alpha_k = self.fixed_alpha
            elif stepsize_type == 'diminishing':
                alpha_k = beta / (gamma + k)
            else:
                obj_value = self.F(w)
                gap = obj_value - self.F_star
                if gap < 2 * current_F_alpha:
                    r += 1
                    k_r = k + 1
                    halving_points.append(k_r)
                    current_alpha = max(current_alpha / 2, min_alpha)
                    current_F_alpha = (current_alpha * self.L * self.M) / (2 * self.mu)
                alpha_k = current_alpha

            self.w = w
            g_k = self.mini_batch_grad() if self.batch_size > 1 else self.stochastic_grad()
            w -= alpha_k * g_k

            obj_history.append(self.F(w))
            grad_norm_history.append(np.linalg.norm(self.grad_F(w)) ** 2)
            dist_to_opt_history.append(np.linalg.norm(w - self.w_star) ** 2)

        return w, np.array(obj_history), np.array(grad_norm_history), np.array(dist_to_opt_history)

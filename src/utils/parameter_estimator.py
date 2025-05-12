import numpy as np

class ParameterEstimator:
    def __init__(self, X, y, model, noise=0.01):
        self.model = model
        self.X = model.X
        self.y = model.y
        self.noise = noise
        self.w_init = model.initialize_weights()
        self.w_star = model.w_star
        
    def compute_L(self, num_samples=1000):
        """
        Estimates Lipschitz constant L.

        Args:
            num_samples: Samples for estimating Lipschitz constant.

        Returns:
            Lipschitz constant.
        """
        n = self.w_init.shape[0]
        L_vals = []
        for _ in range(num_samples):
            w1, w2 = np.random.randn(n), np.random.randn(n)
            g1 = self.model.grad_F(w1)
            g2 = self.model.grad_F(w2)
            if np.linalg.norm(w1 - w2) > 1e-8:
                L_vals.append(np.linalg.norm(g1 - g2) / np.linalg.norm(w1 - w2))
        return max(L_vals) if L_vals else 1.
    
    def compute_c(self):
        """
        Computes strong convexity constant c.

        Returns:
            Minimum eigenvalue of Hessian approximation.
        """
        H = (1 / self.X.shape[0]) * self.X.T @ self.X
        eigenvalues = np.linalg.eigvalsh(H)
        return max(min(eigenvalues), 1e-6)

    def estimate_parameters(self):
        """
        Estimates gradient variance and smoothness parameters.

        Returns:
            Tuple of constants (mu, mu_G, M, M_V, M_G).
        """
        L = self.compute_L()
        c = self.compute_c()
        mu = 1
        mu_G = 1
        sigma2 = self.noise ** 2
        x_norms_squared = np.sum(self.X ** 2, axis=1)
        E_x_norm_sq = np.mean(x_norms_squared)
        x_norms_fourth = x_norms_squared ** 2
        E_x_norm_fourth = np.mean(x_norms_fourth)
        M = sigma2 * E_x_norm_sq
        w_diff = self.w_init - self.w_star
        E_w_minus_A_sq = np.mean(w_diff ** 2)
        M_V = E_w_minus_A_sq * E_x_norm_fourth
        M_G = M_V + mu_G ** 2
        return dict(L=L, c=c, mu=mu, mu_G=mu_G, M=M, M_V=M_V, M_G=M_G)
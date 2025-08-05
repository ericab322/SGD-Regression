import numpy as np

class ParameterEstimator:
    def __init__(self, X, y, model, noise=0.01):
        self.model = model
        self.X = model.X
        self.y = model.y
        self.noise = noise
        self.w_init = model.initialize_weights()
        self.w_star = model.w_star
        
    def compute_empirical_noise(self):
        """
        Computes empirical noise variance.
        """
        residuals = self.y - self.X @ self.w_star
        return np.var(residuals)
    
    def empirical_moments(self, w=None):
        """
        Computes empirical moments for gradient variance and smoothness.
        If w is None, uses initialized weights.
        """
        if w is None:
            w = self.w_init
        full_grad = self.model.grad_F(w)
        grad_norm = np.linalg.norm(full_grad)

        per_sample_grads = np.array([self.model.grad_f_i(w, i) for i in range(self.model.m)])
        mean_grad = np.mean(per_sample_grads, axis=0)

        mu = (full_grad @ mean_grad) / (grad_norm**2 + 1e-12)
        mu_G = np.linalg.norm(mean_grad) / (grad_norm + 1e-12)

        var_total = np.mean(np.sum((per_sample_grads - mean_grad)**2, axis=1))
        M_V = max(0, var_total / (grad_norm**2 + 1e-12))
        M = var_total - M_V * grad_norm**2
        M_G = M_V + mu_G**2

        return dict(mu=mu, mu_G=mu_G, M=M, M_V=M_V, M_G=M_G)


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

    def estimate_parameters(self,use_empirical_noise=True):
        """
        Estimates gradient variance and smoothness parameters.

        Returns:
            Tuple of constants (mu, mu_G, M, M_V, M_G).
        """
        L = self.compute_L()
        c = self.compute_c()

        if use_empirical_noise:
            sigma2 = self.compute_empirical_noise()
        else:
            sigma2 = self.noise ** 2  # for synthetic experiments

        moment_stats = self.empirical_moments(w=self.w_init)
        moment_stats['M'] = sigma2 * np.mean(np.sum(self.X ** 2, axis=1)) 
        return dict(L=L, c=c, **moment_stats)
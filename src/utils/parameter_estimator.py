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


    def compute_L(self, eps=1e-4, trials=50, quantile=0.95,
              safety_factor=2.0, min_L=1e-6, max_L=1e6):
        """
        Estimates Lipschitz constant L for gradient of objective F.
        Works for polynomial regression and neural networks.

        eps: perturbation size for empirical method.
        trials: number of perturbations for empirical method.
        quantile: quantile of observed local Lipschitz ratios.
        safety_factor: multiplier for stability.
        min_L, max_L: clamps for stability.
        """

        def spectral_from_X(X):
            # Spectral norm of Hessian: (1/m) X^T X
            m = X.shape[0]
            sigma_max = np.linalg.svd(X, compute_uv=False)[0]
            return (sigma_max ** 2) / m

        def empirical_local(model, w_ref):
            g_ref = model.grad_F(w_ref)
            ratios = []
            for _ in range(trials):
                u = np.random.randn(*w_ref.shape)
                u /= (np.linalg.norm(u) + 1e-12)
                u *= eps
                g2 = model.grad_F(w_ref + u)
                ratio = np.linalg.norm(g2 - g_ref) / (np.linalg.norm(u) + 1e-12)
                ratios.append(ratio)
            return float(np.quantile(np.array(ratios), quantile))

        # --- Detect model type ---
        if hasattr(self.model, "X") and hasattr(self.model, "w_star"):
            # Likely polynomial regression
            L_est = spectral_from_X(self.model.X)
        else:
            # Likely neural network or other nonlinear model
            w_ref = self.model.initialize_weights()
            L_est = empirical_local(self.model, w_ref)

        # --- Safety clamps ---
        L_est = max(min_L, min(L_est, max_L))
        return L_est * safety_factor

    
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
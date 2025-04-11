import numpy as np



class SGD:
    """""
    Stochastic Gradient Descent for Linear and Polynomial Regression

    Supports arbitrary data sampling strategies. Handles regression on data generated
    from a linear or polynomial model. User specifies the polynomial degree; degree=1
    corresponds to linear regression. 
    
        Model (Linear Case):
            - Prediction: ŷ_i = w_0 + w_1 x_i1 + ... + w_n x_in
            - Loss: f_i(w) = (1/2) (w^T x_{i,new} - y_i)^2
            - Objective: F(w) = (1/m) Σ f_i(w)
            - Gradient: ∇f_i(w) = (w^T x_{i,new} - y_i) x_{i,new}
        
        Model (Polynomial Case):
            - Prediction: ŷ_i = w_0 + w_1 x_i1 + w_2 x_i1^2
    """""
    def __init__(self, X, y, num_iterations=1000, noise=0.01, degree=1):
        """
        Initializes the SGD model with generated data. 

        Args:
            X: The input data matrix of shape (m, n), where m is the number of samples 
                and n is the number of features.
            y: The output data vector of shape (m,).
            num_iterations: The number of iterations for running the SGD optimization. Defaults to 1000.
            noise: The standard deviation of the noise added to the output. Defaults to 0.001.
            degree: The degree of the polynomial features. Defaults to 1 (linear regression).
        """
        self.X = X
        self.y = y.flatten()
        self.m, self.n = X.shape
        self.degree = degree
        self.noise = noise
        self.num_iterations = num_iterations

        self.X_model = self.generate_polynomial_features(self.X, degree)
        self.n_model = self.X_model.shape[1]
        
        self.w = np.zeros(self.n_model)
        self.w_star = np.linalg.solve(self.X_model.T @ self.X_model) @ self.X_model.T @ self.y
        self.F_star = self.F(self.w_star)

        self.L = self.compute_L()
        self.c = self.compute_c()
        self.mu, self.mu_G, self.M, self.M_V, self.M_G = self.estimate_parameters()
        self.fixed_alpha = self.compute_fixed_stepsize()
        self.beta, self.gamma = self.compute_diminishing_stepsize()
    
    def generate_polynomial_features(self, X, degree):
        """
        Generates polynomial features up to the specified degree.
        
        Args:
            X: The input data matrix.
            degree: The degree of the polynomial features.

        Returns:
            X_poly: The polynomial feature matrix.
        """
        X_poly = [np.ones((X.shape[0], 1))] 
        for d in range(1, degree + 1):
            X_poly.append(X ** d)
        return np.hstack(X_poly)
        
    def f_i(self, w, i):
        """
        Computes the loss for a single sample.

        Args:
            w: The parameter vector.
            i: The index of the current training sample.

        Returns:
            he loss for the i-th training sample.
        """
        return 0.5 * (self.X_model[i] @ w - self.y[i]) ** 2
    
    def grad_f_i(self, w, i):
        """
        Computes the gradient of the loss function with respect to parameters for a single sample.

        Args:
            w: The parameter vector.
            i: The index of the current training sample.

        Returns:
            The gradient of the loss function for the i-th sample.
        """
        return (self.X_model[i] @ w - self.y[i]) * self.X_model[i]
    

    def F(self, w):
        """
        Computes the average loss over all samples.

        Args:
            w: The parameter vector.

        Returns:
            float: The average loss over all samples.
        """
        F = (1/self.m) * sum(self.f_i(w, j) for j in range(self.m))
        return F

    def grad_F(self, w):
        """
        Computes the gradient of the objective function with respect to the parameters.

        Args:
            w: The parameter vector.

        Returns:
            The gradient of the objective function.
        """
        grad_F = (1/self.m) * sum(self.grad_f_i(w, i) for i in range(self.m))
        return grad_F
    
    def stochastic_grad(self):
        """
        Computes the stochastic gradient (using a random training sample).

        Returns:
            The stochastic gradient based on a randomly selected training sample.
        """
        i = np.random.randint(0, self.m)
        grad = self.grad_f_i(self.w, i)
        return grad 
    
    def mini_batch_grad(self):
        """
        Computes the mini-batch gradient (using a random subset of training samples).

        Args:
            batch_size: The number of samples in the mini-batch.

        Returns:
            The mini-batch gradient.
        """
        indices = np.random.choice(self.m, self.batch_size, replace=False)
        return (1 / self.batch_size) * sum(self.grad_f_i(self.w, i) for i in indices)
    
    def compute_L(self, num_samples=1000):
        """
        Computes the Lipschitz constant L of the gradient of the objective function.

        Args:
            num_samples: The number of random samples to estimate L. Default is 1000.

        Returns:
            The estimated Lipschitz constant.
        """
        L_vals = []
        d = self.X_model.shape[1]
        for _ in range(num_samples):
            w1, w2 = np.random.randn(d), np.random.randn(d)
            grad_diff = np.linalg.norm(self.grad_F(w1) - self.grad_F(w2), 2)
            w_diff = np.linalg.norm(w1 - w2, 2)
            
            if w_diff > 1e-8: 
                L_vals.append(grad_diff / w_diff)
        return max(L_vals) if L_vals else 1.0
    
    def compute_c(self):
        """
        Computes the constant c associated with strong convexity (Assumption 4.5).

        Args:
            num_samples (int): The number of random samples to estimate L. Default is 1000.

        Returns:
            float: The constant c.
        """
        H = (1/self.m) * (self.X_model.T @ self.X_model)
        eigenvalues = np.linalg.eigvalsh(H)
        c = max(min(eigenvalues), 1e-6) 
        return c
    
    def estimate_parameters(self, num_samples=200):
        """
        Estimates the parameters bounding the variance of the gradient updates.

        Returns:
            tuple: Estimated parameters (mu, mu_G, M, M_V, M_G).
        """
        mu = 1 
        mu_G = 1
        sigma2 = self.noise ** 2
        x_norms_squared = np.sum(self.X_model ** 2, axis=1)
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
        Computes the fixed stepsize for SGD using estimated parameters.

        Returns:
            float: The computed fixed stepsize for the SGD algorithm.
        """
        fixed_alpha = (self.mu / (self.L * self.M_G))
        return fixed_alpha
    
    def compute_halving_stepsize(self):
        """
        Halves stepsize for SGD using estimated parameters.
        
        Returns:
            float: The necessary parameters for computing the havling stepsize in optimize method.

        """
        alpha = self.fixed_alpha
        F_alpha = (alpha * self.L * self.M) / (2 * self.c * self.mu)
        min_alpha = 1e-5
        return alpha, F_alpha, min_alpha
    # figure out when to halve (dynamically look at when gradient stops changing enough)
    
    def compute_diminishing_stepsize(self):
        gamma = ((self.L * self.M_G) / (self.c * self.mu))
        beta = (1 / (self.c * self.mu)) + (self.mu / (self.L * self.M_G))
        return beta, gamma
    
    def optimize(self, stepsize_type='fixed', batch_size=1):
        """
        Runs the SGD optimization process for a specified number of iterations.

        Args:
            stepsize_type: 'fixed', 'halving', or 'diminishing' stepsize.

        Returns:
            tuple: Optimized parameters, the history of the objective function, gradient norms, and 
                   distance to the optimal solution.
        """
        self.batch_size = batch_size
        self.w = np.zeros(self.n_model)
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
        grad_norm_history = [np.linalg.norm(self.grad_F(w))**2]
        dist_to_opt_history = [np.linalg.norm(w - self.w_star)**2]
        
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
                    r+=1
                    k_r = k + 1
                    halving_points.append(k_r)
                    current_alpha = max(current_alpha / 2, min_alpha)
                    current_F_alpha = (current_alpha * self.L * self.M) / (2 * self.mu)  
                alpha_k = current_alpha
            self.w = w
            if self.batch_size > 1:
                g_k = self.mini_batch_grad()
            else:
                g_k = self.stochastic_grad()
            w -= alpha_k * g_k
            obj_history.append(self.F(w))
            grad_norm_history.append(np.linalg.norm(self.grad_F(w))**2)
            dist_to_opt_history.append(np.linalg.norm(w - self.w_star)**2)
        return w, np.array(obj_history), np.array(grad_norm_history), np.array(dist_to_opt_history)
    
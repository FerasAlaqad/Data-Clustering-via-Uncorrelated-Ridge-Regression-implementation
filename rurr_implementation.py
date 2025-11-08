
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment


class RURR_SL:
    """
    Rescaled Uncorrelated Ridge Regression with Soft Label (RURR-SL)
    Automatic optimal scaling parameter
    """
    
    def __init__(self, n_clusters, lambda_reg=1.0, max_iter=100, tol=1e-6):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        lambda_reg : float
            Regularization parameter λ
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.n_clusters = n_clusters
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.Z = None
        self.Y = None
        self.alpha = None
        self.b = None
        self.objective_history = []
        
    def _compute_total_scatter(self, X):
        """Compute total scatter matrix S_t = XHX^T + λI"""
        n = X.shape[1]
        H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
        S_t = X @ H @ X.T + self.lambda_reg * np.eye(X.shape[0])
        return S_t, H
    
    def _initialize_Y(self, n_samples):
        """Initialize soft label matrix Y randomly"""
        Y = np.random.rand(n_samples, self.n_clusters)
        Y = Y / Y.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
        return Y
    
    def _update_Z(self, X, Y, S_t, H):
        """
        Update Z using SVD (Theorem 1)
        Z = S_t^(-1/2) U V^T where M = U S V^T
        Uses eigendecomposition for numerical stability
        """
        # Compute S_t^(-1/2) using eigendecomposition (more stable than sqrtm(inv(S_t)))
        eigvals, eigvecs = np.linalg.eigh(S_t)
        eigvals = np.maximum(eigvals, 1e-10)  # Clip small eigenvalues for stability
        S_t_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Compute M = S_t^(-1/2) X H Y
        M = S_t_inv_sqrt @ X @ H @ Y
        
        # Compact SVD
        U, S, Vt = svd(M, full_matrices=False)
        
        # Z = S_t^(-1/2) U V^T
        Z = S_t_inv_sqrt @ U @ Vt
        
        return Z
    
    def _update_alpha(self, Z, X, Y, H):
        """
        Update scaling parameter α
        α = Tr(Z^T X H Y) / Tr(Y^T H Y)
        """
        numerator = np.trace(Z.T @ X @ H @ Y)
        denominator = np.trace(Y.T @ H @ Y)
        alpha = numerator / (denominator + 1e-10)
        return alpha
    
    def _update_b(self, Z, X, Y, alpha):
        """
        Update bias b
        b = (1/n)(αY^T - Z^T X)1_n
        """
        n = X.shape[1]
        ones_n = np.ones((n, 1))
        b = (1/n) * (alpha * Y.T - Z.T @ X) @ ones_n
        return b
    
    def _newton_method_sigma(self, p_i, max_iter_newton=20):
        """
        Solve for σ̄ using Newton method as in paper's equation (19)
        f(σ̄) = (1/c) * Σ(σ̄ - p_ij)_+ - σ̄ = 0
        Paper's Algorithm 1 line 11: Update σ̄ via Newton method in (20)
        """
        c = len(p_i)
        sigma_bar = np.mean(p_i)  # Initial guess
        
        for _ in range(max_iter_newton):
            # f(σ̄) = (1/c) * Σ(σ̄ - p_ij)_+ - σ̄
            diff = sigma_bar - p_i
            f = np.mean(np.maximum(diff, 0)) - sigma_bar
            
            # f'(σ̄) = (1/c) * Σ1{σ̄ > p_ij} - 1
            f_prime = np.mean((diff > 0).astype(float)) - 1
            
            if abs(f_prime) < 1e-10:
                break
                
            # Newton update: σ̄_{t+1} = σ̄_t - f(σ̄_t) / f'(σ̄_t)
            sigma_bar_new = sigma_bar - f / f_prime
            
            if abs(sigma_bar_new - sigma_bar) < 1e-6:
                break
                
            sigma_bar = sigma_bar_new
            
        return sigma_bar
    
    def _update_Y(self, X, Z, b, alpha):
        """
        Update soft label Y
        Paper's Algorithm 1 (line 494-498): p_i = v_i + (α/c)*1_c - (1^T_c*v_i/c)*1_c
        Paper's equation (17): y^(α)_ij = (p_ij - σ̄)_+
        Paper's Algorithm 1 (line 505-507): Y = (1/α) * Y^(α)
        """
        n = X.shape[1]
        V = X.T @ Z + np.ones((n, 1)) @ b.T
        
        Y_alpha = np.zeros((n, self.n_clusters))
        
        for i in range(n):
            v_i = V[i, :]
            
            # Compute p_i as in paper's Algorithm 1 (line 494-498)
            p_i = v_i + (alpha / self.n_clusters) - np.mean(v_i)
            
            # Solve for σ̄ using Newton method (equation 19)
            sigma_bar = self._newton_method_sigma(p_i)
            
            # Update y_i^(α) = (p_i - σ̄)_+ as in equation (17)
            Y_alpha[i, :] = np.maximum(p_i - sigma_bar, 0)
        
        # Y = (1/α) * Y^(α) as in Algorithm 1 line 16
        Y = Y_alpha / (alpha + 1e-10)
        
        # No additional normalization - Algorithm 1 doesn't normalize after line 16
        return Y
    
    def _compute_objective(self, X, Z, b, Y, alpha, H):
        """Compute objective function value"""
        n = X.shape[1]
        # Paper's formulation (Eq. 7): ||X^T Z + 1_n b^T - αY||_F^2 + λ||Z||_F^2
        term1 = np.linalg.norm(X.T @ Z + np.ones((n, 1)) @ b.T - alpha * Y, 'fro')**2
        term2 = self.lambda_reg * np.linalg.norm(Z, 'fro')**2
        return term1 + term2
    
    def fit(self, X):
        """
        Fit the RURR-SL model
        
        Parameters:
        -----------
        X : array-like, shape (d, n)
            Data matrix where d is dimension and n is number of samples
        """
        d, n = X.shape
        
        # Compute total scatter matrix
        S_t, H = self._compute_total_scatter(X)
        
        # Initialize Y
        self.Y = self._initialize_Y(n)
        self.alpha = 1.0
        
        for iteration in range(self.max_iter):
            # Update Z
            self.Z = self._update_Z(X, self.Y, S_t, H)
            
            # Update alpha
            self.alpha = self._update_alpha(self.Z, X, self.Y, H)
            
            # Update b
            self.b = self._update_b(self.Z, X, self.Y, self.alpha)
            
            # Update Y
            Y_new = self._update_Y(X, self.Z, self.b, self.alpha)
            
            # Check convergence
            change = np.linalg.norm(Y_new - self.Y, 'fro')
            self.Y = Y_new
            
            # Compute objective
            obj = self._compute_objective(X, self.Z, self.b, self.Y, self.alpha, H)
            self.objective_history.append(obj)
            
            if change < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def predict(self):
        """Get cluster assignments from soft labels"""
        return np.argmax(self.Y, axis=1)
    
    def get_soft_labels(self):
        """Get soft label matrix"""
        return self.Y


class URR_SL:
    """
    Uncorrelated Ridge Regression with Soft Label (URR-SL)
    Requires manual scaling parameter α
    """
    
    def __init__(self, n_clusters, alpha=1.0, lambda_reg=1.0, max_iter=100, tol=1e-6):
        """
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        alpha : float
            Fixed scaling parameter
        lambda_reg : float
            Regularization parameter λ
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        self.n_clusters = n_clusters
        self.alpha = alpha  # Fixed alpha
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.Z = None
        self.Y = None
        self.b = None
        self.objective_history = []
    
    def _compute_total_scatter(self, X):
        """Compute total scatter matrix S_t = XHX^T + λI"""
        n = X.shape[1]
        H = np.eye(n) - np.ones((n, n)) / n
        S_t = X @ H @ X.T + self.lambda_reg * np.eye(X.shape[0])
        return S_t, H
    
    def _initialize_Y(self, n_samples):
        """Initialize soft label matrix Y randomly"""
        Y = np.random.rand(n_samples, self.n_clusters)
        Y = Y / Y.sum(axis=1, keepdims=True)
        return Y
    
    def _update_Z(self, X, Y, S_t, H):
        """
        Update Z using SVD
        Uses eigendecomposition for numerical stability
        """
        # Compute S_t^(-1/2) using eigendecomposition (more stable than sqrtm(inv(S_t)))
        eigvals, eigvecs = np.linalg.eigh(S_t)
        eigvals = np.maximum(eigvals, 1e-10)  # Clip small eigenvalues for stability
        S_t_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Compute M = S_t^(-1/2) X H Y
        M = S_t_inv_sqrt @ X @ H @ Y
        
        # Compact SVD
        U, S, Vt = svd(M, full_matrices=False)
        
        # Z = S_t^(-1/2) U V^T
        Z = S_t_inv_sqrt @ U @ Vt
        
        return Z
    
    def _update_b(self, Z, X, Y):
        """Update bias b"""
        n = X.shape[1]
        ones_n = np.ones((n, 1))
        b = (1/n) * (self.alpha * Y.T - Z.T @ X) @ ones_n
        return b
    
    def _newton_method_sigma(self, p_i, max_iter_newton=20):
        """
        Solve for σ̄ using Newton method as in paper's equation (19)
        f(σ̄) = (1/c) * Σ(σ̄ - p_ij)_+ - σ̄ = 0
        Paper's Algorithm 1 line 11: Update σ̄ via Newton method in (20)
        """
        c = len(p_i)
        sigma_bar = np.mean(p_i)  # Initial guess
        
        for _ in range(max_iter_newton):
            # f(σ̄) = (1/c) * Σ(σ̄ - p_ij)_+ - σ̄
            diff = sigma_bar - p_i
            f = np.mean(np.maximum(diff, 0)) - sigma_bar
            
            # f'(σ̄) = (1/c) * Σ1{σ̄ > p_ij} - 1
            f_prime = np.mean((diff > 0).astype(float)) - 1
            
            if abs(f_prime) < 1e-10:
                break
                
            # Newton update: σ̄_{t+1} = σ̄_t - f(σ̄_t) / f'(σ̄_t)
            sigma_bar_new = sigma_bar - f / f_prime
            
            if abs(sigma_bar_new - sigma_bar) < 1e-6:
                break
                
            sigma_bar = sigma_bar_new
            
        return sigma_bar
    
    def _update_Y(self, X, Z, b):
        """
        Update soft label Y
        Paper's Algorithm 1 (line 494-498): p_i = v_i + (α/c)*1_c - (1^T_c*v_i/c)*1_c
        Paper's equation (17): y^(α)_ij = (p_ij - σ̄)_+
        Paper's Algorithm 1 (line 505-507): Y = (1/α) * Y^(α)
        """
        n = X.shape[1]
        V = X.T @ Z + np.ones((n, 1)) @ b.T
        
        Y_alpha = np.zeros((n, self.n_clusters))
        
        for i in range(n):
            v_i = V[i, :]
            
            # Compute p_i as in paper's Algorithm 1 (line 494-498)
            p_i = v_i + (self.alpha / self.n_clusters) - np.mean(v_i)
            
            # Solve for σ̄ using Newton method
            sigma_bar = self._newton_method_sigma(p_i)
            
            # Update y_i^(α) = (p_i - σ̄)_+ as in equation (17)
            Y_alpha[i, :] = np.maximum(p_i - sigma_bar, 0)
        
        # Y = (1/α) * Y^(α) as in Algorithm 1 line 16
        Y = Y_alpha / (self.alpha + 1e-10)
        
        # No additional normalization - Algorithm 1 doesn't normalize after line 16
        return Y
    
    def _compute_objective(self, X, Z, b, Y, H):
        """Compute objective function value"""
        n = X.shape[1]
        # Paper's formulation: ||X^T Z + 1_n b^T - αY||_F^2 + λ||Z||_F^2
        term1 = np.linalg.norm(X.T @ Z + np.ones((n, 1)) @ b.T - self.alpha * Y, 'fro')**2
        term2 = self.lambda_reg * np.linalg.norm(Z, 'fro')**2
        return term1 + term2
    
    def fit(self, X):
        """Fit the URR-SL model"""
        d, n = X.shape
        S_t, H = self._compute_total_scatter(X)
        self.Y = self._initialize_Y(n)
        
        for iteration in range(self.max_iter):
            self.Z = self._update_Z(X, self.Y, S_t, H)
            self.b = self._update_b(self.Z, X, self.Y)
            Y_new = self._update_Y(X, self.Z, self.b)
            
            change = np.linalg.norm(Y_new - self.Y, 'fro')
            self.Y = Y_new
            
            obj = self._compute_objective(X, self.Z, self.b, self.Y, H)
            self.objective_history.append(obj)
            
            if change < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break
        
        return self
    
    def predict(self):
        """Get cluster assignments"""
        return np.argmax(self.Y, axis=1)
    
    def get_soft_labels(self):
        """Get soft label matrix"""
        return self.Y


def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    # Build confusion matrix
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    
    # Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    accuracy = sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) / len(y_true)
    return accuracy

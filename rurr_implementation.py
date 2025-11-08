
import numpy as np
from scipy.linalg import svd, sqrtm
from scipy.optimize import linear_sum_assignment


class RURR_SL:
    """
    Rescaled Uncorrelated Ridge Regression with Soft Label (RURR-SL)
    
    Paper: Algorithm 1, Equation (6)
    Objective: min_{Z,α,b,Y} ||X^T Z + 1_n b^T - αY||_F^2 + λ||Z||_F^2
    Subject to: Z^T S_t Z = I, Y1_c = 1_n, Y ≥ 0
    """
    
    def __init__(self, n_clusters, lambda_reg=1.0, max_iter=100, tol=1e-6, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int (c in paper)
            Number of clusters
        lambda_reg : float (λ in paper)
            Regularization parameter
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.Z = None
        self.Y = None
        self.alpha = None
        self.b = None
        self.objective_history = []
        
    def _compute_total_scatter(self, X):
        """
        Paper: S_t = XHX^T + λI (before Equation 4)
        Paper: H = I - (1/n)1_n1_n^T (Section III)
        """
        n = X.shape[1]
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))
        S_t = X @ H @ X.T + self.lambda_reg * np.eye(X.shape[0])
        return S_t, H
    
    def _initialize_Y(self, n_samples):
        """
        Paper Algorithm 1, line 1: "Initialize random soft matrix Y satisfying Y1_c = 1_n"
        """
        rng = np.random.RandomState(self.random_state)
        Y = rng.rand(n_samples, self.n_clusters)
        Y = Y / Y.sum(axis=1, keepdims=True)
        return Y
    
    def _update_Z(self, X, Y, S_t, H):
        """
        Paper Algorithm 1, lines 3-5
        Paper Theorem 1, Equation (29): Z = S_t^{-1/2} U V^T
        Paper Equation (27): M = S_t^{-1/2} X H Y
        """
        # Algorithm 1, line 3: Update M ← S_t^{-1/2} X H Y
        S_t_inv_sqrt = sqrtm(np.linalg.inv(S_t))
        M = S_t_inv_sqrt @ X @ H @ Y
        
        # Algorithm 1, line 4: Calculate U S V^T = M via compact SVD
        U, S, Vt = svd(M, full_matrices=False)
        
        # Algorithm 1, line 5: Update Z ← S_t^{-1/2} U V^T
        Z = S_t_inv_sqrt @ U @ Vt
        
        return Z
    
    def _update_alpha(self, Z, X, Y, H):
        """
        Paper Equation (25): α = Tr(Z^T X H Y) / Tr(Y^T H Y)
        Paper Algorithm 1, line 6
        """
        numerator = np.trace(Z.T @ X @ H @ Y)
        denominator = np.trace(Y.T @ H @ Y)
        alpha = numerator / denominator
        return alpha
    
    def _update_b(self, Z, X, Y, alpha):
        """
        Paper Equation (22): b = (1/n)(αY^T - Z^T X)1_n
        Paper Algorithm 1, line 7
        """
        n = X.shape[1]
        ones_n = np.ones((n, 1))
        b = (1.0 / n) * (alpha * Y.T - Z.T @ X) @ ones_n
        return b
    
    def _newton_method_sigma(self, p_i, max_iter_newton=20):
        """
        Paper Equation (19): f(σ̄) = (1/c)Σ(σ̄ - p_ij)_+ - σ̄ = 0
        Paper Equation (20): σ̄_{t+1} = σ̄_t - f(σ̄_t)/f'(σ̄_t)
        """
        c = len(p_i)
        sigma_bar = np.mean(p_i)
        
        for _ in range(max_iter_newton):
            diff = sigma_bar - p_i
            f = (1.0 / c) * np.sum(np.maximum(diff, 0)) - sigma_bar
            f_prime = (1.0 / c) * np.sum((diff > 0).astype(float)) - 1.0
            
            if abs(f_prime) < 1e-10:
                break
            
            sigma_bar_new = sigma_bar - f / f_prime
            
            if abs(sigma_bar_new - sigma_bar) < 1e-6:
                break
            
            sigma_bar = sigma_bar_new
        
        return sigma_bar
    
    def _update_Y(self, X, Z, b, alpha):
        """
        Paper Algorithm 1, lines 8-16
        Paper Equation (17): y^{(α)}_{ij} = (p_{ij} - σ̄)_+
        """
        n = X.shape[1]
        # Algorithm 1, line 8: Update V ← X^T Z + 1_n b^T
        V = X.T @ Z + np.ones((n, 1)) @ b.T
        
        Y_alpha = np.zeros((n, self.n_clusters))
        
        # Algorithm 1, line 9: for i = 1 : n do
        for i in range(n):
            v_i = V[i, :]
            
            # Algorithm 1, line 10: Update p_i ← v_i + (α/c)1_c - (1_c^T v_i/c)1_c
            p_i = v_i + (alpha / self.n_clusters) * np.ones(self.n_clusters) - (np.sum(v_i) / self.n_clusters) * np.ones(self.n_clusters)
            
            # Algorithm 1, line 11: Update σ̄ via Newton method
            sigma_bar = self._newton_method_sigma(p_i)
            
            # Algorithm 1, lines 12-14: for j = 1 : c do
            # Paper Equation (17): y^{(α)}_{ij} = (p_{ij} - σ̄)_+
            for j in range(self.n_clusters):
                Y_alpha[i, j] = max(p_i[j] - sigma_bar, 0)
        
        # Algorithm 1, line 16: Calculate Y = (1/α)Y^{(α)}
        Y = Y_alpha / alpha
        
        return Y
    
    def _compute_objective(self, X, Z, b, Y, alpha):
        """
        Paper Equation (6): ||X^T Z + 1_n b^T - αY||_F^2 + λ||Z||_F^2
        """
        n = X.shape[1]
        term1 = np.linalg.norm(X.T @ Z + np.ones((n, 1)) @ b.T - alpha * Y, 'fro') ** 2
        term2 = self.lambda_reg * np.linalg.norm(Z, 'fro') ** 2
        return term1 + term2
    
    def fit(self, X):
        """
        Fit RURR-SL model following Algorithm 1
        
        Parameters
        ----------
        X : ndarray of shape (d, n)
            Data matrix where d is dimension and n is number of samples
        """
        d, n = X.shape
        
        S_t, H = self._compute_total_scatter(X)
        
        # Algorithm 1, line 1: Initialize Y
        self.Y = self._initialize_Y(n)
        self.alpha = 1.0
        
        # Algorithm 1, line 2: while not converge do
        for iteration in range(self.max_iter):
            Y_old = self.Y.copy()
            
            # Lines 3-5: Update Z
            self.Z = self._update_Z(X, self.Y, S_t, H)
            
            # Line 6: Update α
            self.alpha = self._update_alpha(self.Z, X, self.Y, H)
            
            # Line 7: Update b
            self.b = self._update_b(self.Z, X, self.Y, self.alpha)
            
            # Lines 8-16: Update Y
            self.Y = self._update_Y(X, self.Z, self.b, self.alpha)
            
            # Check convergence
            change = np.linalg.norm(self.Y - Y_old, 'fro')
            
            # Compute objective
            obj = self._compute_objective(X, self.Z, self.b, self.Y, self.alpha)
            self.objective_history.append(obj)
            
            if change < self.tol:
                break
        
        return self
    
    def predict(self):
        """Get cluster assignments"""
        return np.argmax(self.Y, axis=1)
    
    def get_soft_labels(self):
        """Get soft label matrix"""
        return self.Y


class URR_SL:
    """
    Uncorrelated Ridge Regression with Soft Label (URR-SL)
    Paper Equation (4) with fixed scaling parameter α
    """
    
    def __init__(self, n_clusters, alpha=1.0, lambda_reg=1.0, max_iter=100, tol=1e-6, random_state=None):
        """
        Parameters
        ----------
        n_clusters : int (c in paper)
            Number of clusters
        alpha : float (α in paper)
            Fixed scaling parameter
        lambda_reg : float (λ in paper)
            Regularization parameter
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        random_state : int or None
            Random seed
        """
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.Z = None
        self.Y = None
        self.b = None
        self.objective_history = []
    
    def _compute_total_scatter(self, X):
        """Compute S_t = XHX^T + λI"""
        n = X.shape[1]
        H = np.eye(n) - (1.0 / n) * np.ones((n, n))
        S_t = X @ H @ X.T + self.lambda_reg * np.eye(X.shape[0])
        return S_t, H
    
    def _initialize_Y(self, n_samples):
        """Initialize Y"""
        rng = np.random.RandomState(self.random_state)
        Y = rng.rand(n_samples, self.n_clusters)
        Y = Y / Y.sum(axis=1, keepdims=True)
        return Y
    
    def _update_Z(self, X, Y, S_t, H):
        """Update Z"""
        S_t_inv_sqrt = sqrtm(np.linalg.inv(S_t))
        M = S_t_inv_sqrt @ X @ H @ Y
        U, S, Vt = svd(M, full_matrices=False)
        Z = S_t_inv_sqrt @ U @ Vt
        return Z
    
    def _update_b(self, Z, X, Y):
        """Update b with fixed alpha"""
        n = X.shape[1]
        ones_n = np.ones((n, 1))
        b = (1.0 / n) * (self.alpha * Y.T - Z.T @ X) @ ones_n
        return b
    
    def _newton_method_sigma(self, p_i, max_iter_newton=20):
        """Solve for σ̄"""
        c = len(p_i)
        sigma_bar = np.mean(p_i)
        
        for _ in range(max_iter_newton):
            diff = sigma_bar - p_i
            f = (1.0 / c) * np.sum(np.maximum(diff, 0)) - sigma_bar
            f_prime = (1.0 / c) * np.sum((diff > 0).astype(float)) - 1.0
            
            if abs(f_prime) < 1e-10:
                break
            
            sigma_bar_new = sigma_bar - f / f_prime
            
            if abs(sigma_bar_new - sigma_bar) < 1e-6:
                break
            
            sigma_bar = sigma_bar_new
        
        return sigma_bar
    
    def _update_Y(self, X, Z, b):
        """Update Y with fixed alpha"""
        n = X.shape[1]
        V = X.T @ Z + np.ones((n, 1)) @ b.T
        
        Y_alpha = np.zeros((n, self.n_clusters))
        
        for i in range(n):
            v_i = V[i, :]
            p_i = v_i + (self.alpha / self.n_clusters) * np.ones(self.n_clusters) - (np.sum(v_i) / self.n_clusters) * np.ones(self.n_clusters)
            sigma_bar = self._newton_method_sigma(p_i)
            
            for j in range(self.n_clusters):
                Y_alpha[i, j] = max(p_i[j] - sigma_bar, 0)
        
        Y = Y_alpha / self.alpha
        return Y
    
    def _compute_objective(self, X, Z, b, Y):
        """Compute objective"""
        n = X.shape[1]
        term1 = np.linalg.norm(X.T @ Z + np.ones((n, 1)) @ b.T - self.alpha * Y, 'fro') ** 2
        term2 = self.lambda_reg * np.linalg.norm(Z, 'fro') ** 2
        return term1 + term2
    
    def fit(self, X):
        """Fit URR-SL model"""
        d, n = X.shape
        S_t, H = self._compute_total_scatter(X)
        self.Y = self._initialize_Y(n)
        
        for iteration in range(self.max_iter):
            Y_old = self.Y.copy()
            
            self.Z = self._update_Z(X, self.Y, S_t, H)
            self.b = self._update_b(self.Z, X, self.Y)
            self.Y = self._update_Y(X, self.Z, self.b)
            
            change = np.linalg.norm(self.Y - Y_old, 'fro')
            
            obj = self._compute_objective(X, self.Z, self.b, self.Y)
            self.objective_history.append(obj)
            
            if change < self.tol:
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
    Paper Section VI.A
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    
    accuracy = sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) / len(y_true)
    return accuracy


def normalized_mutual_info(y_true, y_pred):
    """
    Compute NMI as defined in Paper Section VI.B
    
    NMI = (Σ_{i,j} n_{ij} log(n_{ij}*n / (n_i * n_j))) / 
          sqrt((Σ_i n_i log(n_i/n)) * (Σ_j n_j log(n_j/n)))
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n = y_true.size
    if n == 0:
        return 0.0
    
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    
    n_true = true_labels.size
    n_pred = pred_labels.size
    
    # Build contingency table
    contingency = np.zeros((n_true, n_pred), dtype=float)
    label_to_index_true = {label: idx for idx, label in enumerate(true_labels)}
    label_to_index_pred = {label: idx for idx, label in enumerate(pred_labels)}
    
    for t, p in zip(y_true, y_pred):
        contingency[label_to_index_true[t], label_to_index_pred[p]] += 1
    
    # Marginals
    n_i = contingency.sum(axis=1)
    n_j = contingency.sum(axis=0)
    
    # Mutual information
    mi = 0.0
    for i in range(n_true):
        for j in range(n_pred):
            n_ij = contingency[i, j]
            if n_ij > 0:
                mi += (n_ij / n) * np.log((n_ij * n) / (n_i[i] * n_j[j]))
    
    # Entropies
    h_true = 0.0
    for i in range(n_true):
        if n_i[i] > 0:
            p = n_i[i] / n
            h_true -= p * np.log(p)
    
    h_pred = 0.0
    for j in range(n_pred):
        if n_j[j] > 0:
            p = n_j[j] / n
            h_pred -= p * np.log(p)
    
    denom = np.sqrt(h_true * h_pred)
    if denom == 0:
        return 0.0
    
    return mi / denom
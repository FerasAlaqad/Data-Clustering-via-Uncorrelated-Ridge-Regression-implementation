import numpy as np
from sklearn.cluster import KMeans


class KMeansClustering:
    """
    Wrapper around scikit-learn's KMeans to expose a consistent interface.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default=300
        Maximum number of iterations.
    random_state : int or None, default=None
        RNG seed.
    """

    def __init__(self, n_clusters, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = None
        self.labels_ = None
        self.centroids_ = None

    def fit(self, X):
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray of shape (d, n)
            Data matrix containing n samples of dimension d.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected X with shape (d, n)")

        model = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=10,
            random_state=self.random_state,
        )
        model.fit(X.T)
        self._model = model
        self.labels_ = model.labels_
        self.centroids_ = model.cluster_centers_.T
        return self

    def predict(self):
        """Return hard cluster assignments."""
        if self.labels_ is None:
            raise RuntimeError("Call fit before predict.")
        return self.labels_


class RMKMC:
    """
    Robust k-means clustering with an ℓ₂,₁-type sample weighting strategy.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Convergence tolerance on centroid movement.
    random_state : int or None, default=None
        RNG seed.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """

    def __init__(
        self,
        n_clusters,
        max_iter=100,
        tol=1e-4,
        random_state=None,
        epsilon=1e-8,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon
        self.labels_ = None
        self.centroids_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected X with shape (d, n)")

        d, n = X.shape
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n, self.n_clusters, replace=False)
        centroids = X[:, indices].copy()

        for _ in range(self.max_iter):
            diff = X[:, :, None] - centroids[:, None, :]  # (d, n, k)
            dist_sq = np.sum(diff**2, axis=0)  # (n, k)
            labels = np.argmin(dist_sq, axis=1)
            distances = np.sqrt(
                dist_sq[np.arange(n), labels] + self.epsilon
            )
            weights = 1.0 / (2.0 * distances + self.epsilon)

            centroids_new = centroids.copy()
            for j in range(self.n_clusters):
                mask = labels == j
                if not np.any(mask):
                    centroids_new[:, j] = X[:, rng.integers(n)]
                    continue
                w = weights[mask]
                weighted_sum = (X[:, mask] * w) @ np.ones(w.shape)
                centroids_new[:, j] = weighted_sum / w.sum()

            shift = np.linalg.norm(centroids_new - centroids)
            centroids = centroids_new
            if shift < self.tol:
                break

        self.labels_ = labels
        self.centroids_ = centroids
        return self

    def predict(self):
        if self.labels_ is None:
            raise RuntimeError("Call fit before predict.")
        return self.labels_


class FuzzyKMeans:
    """
    Standard fuzzy c-means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    fuzziness : float, default=2.0
        Fuzziness parameter m (>1).
    max_iter : int, default=150
        Maximum iterations.
    tol : float, default=1e-5
        Convergence tolerance on membership matrix.
    random_state : int or None, default=None
        RNG seed.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """

    def __init__(
        self,
        n_clusters,
        fuzziness=2.5,
        max_iter=150,
        tol=1e-5,
        random_state=None,
        epsilon=1e-8,
    ):
        if fuzziness <= 1.0:
            raise ValueError("fuzziness must be greater than 1.")
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon
        self.membership_ = None
        self.centroids_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected X with shape (d, n)")

        d, n = X.shape
        rng = np.random.default_rng(self.random_state)
        membership = rng.random((n, self.n_clusters))
        membership = membership / membership.sum(axis=1, keepdims=True)

        for _ in range(self.max_iter):
            membership_power = membership ** self.fuzziness  # (n, k)
            denom = membership_power.sum(axis=0) + self.epsilon  # (k,)
            centroids = (X @ membership_power) / denom  # (d, k)

            diff = X[:, :, None] - centroids[:, None, :]
            dist_sq = np.sum(diff**2, axis=0) + self.epsilon  # (n, k)

            inv_dist = dist_sq ** (-1.0 / (self.fuzziness - 1.0))
            membership_new = inv_dist / inv_dist.sum(axis=1, keepdims=True)

            delta = np.linalg.norm(membership_new - membership)
            membership = membership_new
            if delta < self.tol:
                break

        self.membership_ = membership
        self.centroids_ = centroids
        return self

    def predict(self):
        if self.membership_ is None:
            raise RuntimeError("Call fit before predict.")
        return np.argmax(self.membership_, axis=1)


class RSFKM:
    """
    Robust and sparse fuzzy k-means clustering with capped distances.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
        fuzziness : float, default=2.5
        Fuzziness parameter m (>1).
    cap : int, default=30
        Capping threshold (tuned in {10,15,20,25,30,35,40,45,50} as in the paper).
    gamma : float, default=0.0
        Sparsity regularization strength on memberships.
    max_iter : int, default=150
        Maximum iterations.
    tol : float, default=1e-5
        Convergence tolerance on centroids.
    random_state : int or None, default=None
        RNG seed.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """

    def __init__(
        self,
        n_clusters,
        fuzziness=2.5,
        cap=30,
        gamma=0.0,
        max_iter=150,
        tol=1e-5,
        random_state=None,
        epsilon=1e-8,
    ):
        if fuzziness <= 1.0:
            raise ValueError("fuzziness must be greater than 1.")
        self.n_clusters = n_clusters
        self.fuzziness = fuzziness
        if not isinstance(cap, (int, np.integer)):
            raise ValueError("cap must be an integer.")
        self.cap = float(cap)
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.epsilon = epsilon
        self.membership_ = None
        self.centroids_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Expected X with shape (d, n)")

        d, n = X.shape
        rng = np.random.default_rng(self.random_state)
        membership = rng.random((n, self.n_clusters))
        membership = membership / membership.sum(axis=1, keepdims=True)

        centroids = X[:, rng.choice(n, self.n_clusters, replace=False)].copy()

        for _ in range(self.max_iter):
            diff = X[:, :, None] - centroids[:, None, :]
            dist_sq = np.sum(diff**2, axis=0)
            capped = np.minimum(dist_sq, self.cap) + self.epsilon

            inv_dist = capped ** (-1.0 / (self.fuzziness - 1.0))
            membership = inv_dist / inv_dist.sum(axis=1, keepdims=True)

            if self.gamma > 0:
                membership = np.maximum(0.0, membership - self.gamma)
                membership = membership / (
                    membership.sum(axis=1, keepdims=True) + self.epsilon
                )

            membership_power = membership ** self.fuzziness
            denom = membership_power.sum(axis=0) + self.epsilon
            centroids_new = (X @ membership_power) / denom

            shift = np.linalg.norm(centroids_new - centroids)
            centroids = centroids_new
            if shift < self.tol:
                break

        self.membership_ = membership
        self.centroids_ = centroids
        return self

    def predict(self):
        if self.membership_ is None:
            raise RuntimeError("Call fit before predict.")
        return np.argmax(self.membership_, axis=1)


__all__ = [
    "KMeansClustering",
    "RMKMC",
    "FuzzyKMeans",
    "RSFKM",
]


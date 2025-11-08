"""
Generate the two synthetic (toy) datasets used in the RURR-SL paper.

Based on Figures 3 and 4 from the paper:
1. 3-cluster Gaussian distributed data (Fig. 3)
2. Multicluster data (Fig. 4)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_3cluster_gaussian(
    n_samples_per_cluster=100,
    n_features=3,
    cluster_std=0.8,
    random_state=42
):
    """
    Generate 3-cluster Gaussian distributed data as shown in Fig. 3 of the paper.
    
    This creates three well-separated Gaussian clusters in 2D space.
    
    Parameters
    ----------
    n_samples_per_cluster : int
        Number of samples per cluster (default: 100)
    n_features : int
        Number of features/dimensions (default: 2 for visualization)
    cluster_std : float
        Standard deviation of clusters (default: 0.8)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_features, n_samples)
        Data matrix in (d, n) format
    y : ndarray of shape (n_samples,)
        True cluster labels (0, 1, 2)
    """
    rng = np.random.RandomState(random_state)
    
    # Define cluster centers (well-separated in 2D)
    centers = np.array([
        [-3.0, -3.0],  # Cluster 0: bottom-left
        [3.0, -3.0],   # Cluster 1: bottom-right
        [0.0, 3.5],    # Cluster 2: top-center
    ])
    
    n_clusters = len(centers)
    n_samples = n_samples_per_cluster * n_clusters
    
    X_list = []
    y_list = []
    
    for cluster_id, center in enumerate(centers):
        # Generate samples from Gaussian distribution
        cluster_samples = rng.randn(n_samples_per_cluster, n_features) * cluster_std
        # Always generate in base 2D around centers for visualization
        cluster_samples[:, :2] += center
        
        X_list.append(cluster_samples)
        y_list.extend([cluster_id] * n_samples_per_cluster)
    
    X_base = np.vstack(X_list)  # (n, requested_features)
    # Ensure dimensionality satisfies d >= c (important for URR/RURR)
    target_d = max(int(n_features), n_clusters, 2)
    if target_d > X_base.shape[1]:
        extra = rng.randn(X_base.shape[0], target_d - X_base.shape[1]) * (cluster_std * 0.05)
        X_base = np.hstack([X_base, extra])
    elif target_d < X_base.shape[1]:
        X_base = X_base[:, :target_d]

    X = X_base.T  # Shape: (d, n)
    y = np.array(y_list)
    
    return X, y


def generate_multicluster_data(
    n_samples_per_cluster=50,
    n_features=7,
    random_state=42
):
    """
    Generate multicluster data as shown in Fig. 4 of the paper.
    
    This creates a more complex clustering scenario with multiple clusters
    arranged in a specific pattern (appears to be 6-7 clusters from the figure).
    
    Parameters
    ----------
    n_samples_per_cluster : int
        Number of samples per cluster (default: 50)
    n_features : int
        Number of features/dimensions (default: 2)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : ndarray of shape (n_features, n_samples)
        Data matrix in (d, n) format
    y : ndarray of shape (n_samples,)
        True cluster labels
    """
    rng = np.random.RandomState(random_state)
    
    # Define multiple cluster centers based on Fig. 4
    # The figure shows clusters arranged in a circular/scattered pattern
    centers = np.array([
        [-4.0, -2.0],   # Cluster 0
        [-2.0, -4.0],   # Cluster 1
        [2.0, -4.0],    # Cluster 2
        [4.0, -2.0],    # Cluster 3
        [4.0, 2.0],     # Cluster 4
        [0.0, 0.0],     # Cluster 5 (center)
        [-4.0, 2.0],    # Cluster 6
    ])
    
    # Different standard deviations for variety
    cluster_stds = [0.6, 0.7, 0.6, 0.7, 0.6, 0.8, 0.7]
    
    n_clusters = len(centers)
    
    X_list = []
    y_list = []
    
    for cluster_id, (center, std) in enumerate(zip(centers, cluster_stds)):
        # Generate samples from Gaussian distribution
        cluster_samples = rng.randn(n_samples_per_cluster, n_features) * std
        # Base 2D structure for visualization
        cluster_samples[:, :2] += center
        
        X_list.append(cluster_samples)
        y_list.extend([cluster_id] * n_samples_per_cluster)
    
    X_base = np.vstack(X_list)  # (n, requested_features)
    # Ensure dimensionality satisfies d >= c
    target_d = max(int(n_features), n_clusters, 2)
    if target_d > X_base.shape[1]:
        std_mean = np.mean(cluster_stds)
        extra = rng.randn(X_base.shape[0], target_d - X_base.shape[1]) * (std_mean * 0.05)
        X_base = np.hstack([X_base, extra])
    elif target_d < X_base.shape[1]:
        X_base = X_base[:, :target_d]

    X = X_base.T  # Shape: (d, n)
    y = np.array(y_list)
    
    return X, y


def save_toy_datasets(output_dir="datasets"):
    """
    Generate and save both toy datasets to disk.
    
    Parameters
    ----------
    output_dir : str
        Directory to save the datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Generating toy datasets...")
    
    # Generate 3-cluster Gaussian data
    print("\n1. Generating 3-cluster Gaussian data...")
    X_3cluster, y_3cluster = generate_3cluster_gaussian(
        n_samples_per_cluster=100,
        n_features=3,         # ensure d >= c
        cluster_std=0.8,
        random_state=42
    )
    
    # Save to .npz file
    np.savez(
        output_path / "toy_3cluster_gaussian.npz",
        X=X_3cluster,
        y=y_3cluster,
        description="3-cluster Gaussian distributed data from RURR-SL paper Fig. 3"
    )
    print(f"   Saved: {output_path / 'toy_3cluster_gaussian.npz'}")
    print(f"   Shape: X={X_3cluster.shape}, y={y_3cluster.shape}")
    print(f"   Clusters: {len(np.unique(y_3cluster))}")
    
    # Generate multicluster data
    print("\n2. Generating multicluster data...")
    X_multi, y_multi = generate_multicluster_data(
        n_samples_per_cluster=50,
        n_features=7,         # ensure d >= c
        random_state=42
    )
    
    # Save to .npz file
    np.savez(
        output_path / "toy_multicluster.npz",
        X=X_multi,
        y=y_multi,
        description="Multicluster data from RURR-SL paper Fig. 4"
    )
    print(f"   Saved: {output_path / 'toy_multicluster.npz'}")
    print(f"   Shape: X={X_multi.shape}, y={y_multi.shape}")
    print(f"   Clusters: {len(np.unique(y_multi))}")
    
    # Visualize both datasets
    print("\n3. Creating visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 3-cluster Gaussian
    axes[0].scatter(X_3cluster[0, :], X_3cluster[1, :], 
                   c=y_3cluster, cmap='viridis', s=30, alpha=0.7)
    axes[0].set_title('3-Cluster Gaussian Data (Fig. 3)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature 1', fontsize=12)
    axes[0].set_ylabel('Feature 2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Plot multicluster
    axes[1].scatter(X_multi[0, :], X_multi[1, :], 
                   c=y_multi, cmap='tab10', s=30, alpha=0.7)
    axes[1].set_title('Multicluster Data (Fig. 4)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature 1', fontsize=12)
    axes[1].set_ylabel('Feature 2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'toy_datasets_visualization.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path / 'toy_datasets_visualization.png'}")
    
    print("\n✓ All toy datasets generated successfully!")
    
    return X_3cluster, y_3cluster, X_multi, y_multi


def load_toy_dataset(dataset_name, dataset_dir="datasets"):
    """
    Load a toy dataset from disk.
    
    Parameters
    ----------
    dataset_name : str
        Either '3cluster' or 'multicluster'
    dataset_dir : str
        Directory containing the datasets
        
    Returns
    -------
    X : ndarray of shape (d, n)
        Data matrix
    y : ndarray of shape (n,)
        True labels
    """
    dataset_path = Path(dataset_dir)
    
    if dataset_name == '3cluster':
        data = np.load(dataset_path / "toy_3cluster_gaussian.npz")
    elif dataset_name == 'multicluster':
        data = np.load(dataset_path / "toy_multicluster.npz")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use '3cluster' or 'multicluster'")
    
    return data['X'], data['y']


if __name__ == "__main__":
    # Generate and save datasets
    X_3cluster, y_3cluster, X_multi, y_multi = save_toy_datasets()
    
    # Show statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    print("\n3-Cluster Gaussian:")
    print(f"  Samples: {X_3cluster.shape[1]}")
    print(f"  Features: {X_3cluster.shape[0]}")
    print(f"  Clusters: {len(np.unique(y_3cluster))}")
    print(f"  Samples per cluster: {np.bincount(y_3cluster)}")
    
    print("\nMulticluster:")
    print(f"  Samples: {X_multi.shape[1]}")
    print(f"  Features: {X_multi.shape[0]}")
    print(f"  Clusters: {len(np.unique(y_multi))}")
    print(f"  Samples per cluster: {np.bincount(y_multi)}")
    
    print("\n" + "="*60)
    print("\nTo load these datasets in your experiments:")
    print("  from generate_toy_datasets import load_toy_dataset")
    print("  X, y = load_toy_dataset('3cluster')")
    print("  X, y = load_toy_dataset('multicluster')")


"""
Generate synthetic toy datasets as shown in the paper.

Paper Figure 3: 3-cluster Gaussian distributed data
Paper Figure 4: Multicluster data
"""

import numpy as np
from pathlib import Path

from load_datasets import load_toy_dataset


if __name__ == "__main__":
    """Generate and persist toy datasets, then provide summary/visualization."""
    import matplotlib.pyplot as plt

    toy_dataset_keys = ['3cluster', 'multicluster', 'nested_circles', 'moons']

    print("Generating and saving toy datasets to 'datasets/'...")
    print("="*70)

    # Generate/save datasets
    dataset_cache = {}
    for idx, key in enumerate(toy_dataset_keys, start=1):
        print(f"\n{idx}. {key}:", end=" ")
        X, y = load_toy_dataset(
            key,
            use_saved=False,
            save_generated=True,
            allow_generate=True
        )
        dataset_cache[key] = (X, y)
        npz_path = Path("datasets") / f"{key}.npz"
        print(f"saved {X.shape[1]} samples to {npz_path}")
        print(f"   X shape: {X.shape}")
        print(f"   Number of clusters: {len(np.unique(y))}")
        print(f"   Samples per cluster: {np.bincount(y)}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 3-cluster
    X1, y1 = dataset_cache['3cluster']
    for cluster_id in np.unique(y1):
        mask = (y1 == cluster_id)
        axes[0].scatter(X1[0, mask], X1[1, mask], label=f'Cluster {cluster_id}', alpha=0.6)
    axes[0].set_title('3-Cluster Gaussian (Paper Figure 3)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot multicluster
    X2, y2 = dataset_cache['multicluster']
    for cluster_id in np.unique(y2):
        mask = (y2 == cluster_id)
        axes[1].scatter(X2[0, mask], X2[1, mask], label=f'Cluster {cluster_id}', alpha=0.6, s=20)
    axes[1].set_title('Multicluster (Paper Figure 4)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('toy_datasets.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'toy_datasets.png'")
    print("✓ All toy datasets stored as NPZ files in the 'datasets/' directory")
    print("="*70)
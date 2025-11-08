"""
Benchmark Dataset Experiments for RURR-SL
Reproduces Table I results from the paper
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our implementations
from rurr_implementation import RURR_SL, URR_SL, clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def load_benchmark_datasets():
    """
    Load benchmark datasets similar to paper
    For demonstration, we use scikit-learn datasets
    You can add custom loaders for AT&T, COLON, etc.
    """
    datasets = {}
    
    # 1. Digits dataset (similar complexity to paper's datasets)
    print("Loading Digits dataset...")
    digits = load_digits()
    X_digits = digits.data.T  # (64, 1797)
    y_digits = digits.target
    datasets['Digits'] = {
        'X': X_digits,
        'y': y_digits,
        'n_clusters': 10,
        'name': 'Digits (8x8)'
    }
    
    # 2. Olivetti Faces (similar to AT&T faces)
    print("Loading Olivetti Faces...")
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X_faces = faces.data.T  # (4096, 400)
    y_faces = faces.target
    datasets['Faces'] = {
        'X': X_faces,
        'y': y_faces,
        'n_clusters': 40,
        'name': 'Olivetti Faces'
    }
    
    # 3. Synthetic multicluster data (like paper's toy example)
    print("Generating synthetic multicluster data...")
    from sklearn.datasets import make_blobs
    X_synth, y_synth = make_blobs(n_samples=500, n_features=50, 
                                   centers=5, cluster_std=1.5, 
                                   random_state=42)
    datasets['Synthetic'] = {
        'X': X_synth.T,
        'y': y_synth,
        'n_clusters': 5,
        'name': 'Synthetic 5-cluster'
    }
    
    return datasets


def run_experiment(X, y_true, n_clusters, dataset_name, n_runs=10):
    """
    Run clustering experiment with multiple random initializations
    Returns average metrics across runs
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {X.shape}, Clusters: {n_clusters}")
    print(f"{'='*60}")
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.T).T
    
    results = {
        'kmeans': {'acc': [], 'nmi': []},
        'urr': {'acc': [], 'nmi': []},
        'rurr': {'acc': [], 'nmi': []}
    }
    
    # Lambda values to try
    lambda_values = [0.01, 0.1, 1.0, 10.0]
    
    best_lambda_urr = 1.0
    best_lambda_rurr = 1.0
    best_acc_urr = 0
    best_acc_rurr = 0
    
    # Parameter tuning
    print("\nParameter tuning...")
    for lam in lambda_values:
        # URR-SL
        urr_temp = URR_SL(n_clusters=n_clusters, alpha=1.0, lambda_reg=lam, max_iter=50)
        urr_temp.fit(X_scaled)
        acc_temp = clustering_accuracy(y_true, urr_temp.predict())
        if acc_temp > best_acc_urr:
            best_acc_urr = acc_temp
            best_lambda_urr = lam
        
        # RURR-SL
        rurr_temp = RURR_SL(n_clusters=n_clusters, lambda_reg=lam, max_iter=50)
        rurr_temp.fit(X_scaled)
        acc_temp = clustering_accuracy(y_true, rurr_temp.predict())
        if acc_temp > best_acc_rurr:
            best_acc_rurr = acc_temp
            best_lambda_rurr = lam
    
    print(f"Best λ for URR-SL: {best_lambda_urr}")
    print(f"Best λ for RURR-SL: {best_lambda_rurr}")
    
    # Run experiments with best parameters
    print(f"\nRunning {n_runs} experiments...")
    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end='\r')
        
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=run, n_init=10)
        y_pred_km = kmeans.fit_predict(X_scaled.T)
        results['kmeans']['acc'].append(clustering_accuracy(y_true, y_pred_km))
        results['kmeans']['nmi'].append(normalized_mutual_info_score(y_true, y_pred_km))
        
        # URR-SL
        urr = URR_SL(n_clusters=n_clusters, alpha=1.0, 
                     lambda_reg=best_lambda_urr, max_iter=100)
        urr.fit(X_scaled)
        y_pred_urr = urr.predict()
        results['urr']['acc'].append(clustering_accuracy(y_true, y_pred_urr))
        results['urr']['nmi'].append(normalized_mutual_info_score(y_true, y_pred_urr))
        
        # RURR-SL
        rurr = RURR_SL(n_clusters=n_clusters, lambda_reg=best_lambda_rurr, max_iter=100)
        rurr.fit(X_scaled)
        y_pred_rurr = rurr.predict()
        results['rurr']['acc'].append(clustering_accuracy(y_true, y_pred_rurr))
        results['rurr']['nmi'].append(normalized_mutual_info_score(y_true, y_pred_rurr))
    
    print(f"  Run {n_runs}/{n_runs}... Done!")
    
    # Compute statistics
    stats = {}
    for method in results:
        stats[method] = {
            'acc_mean': np.mean(results[method]['acc']),
            'acc_std': np.std(results[method]['acc']),
            'nmi_mean': np.mean(results[method]['nmi']),
            'nmi_std': np.std(results[method]['nmi'])
        }
    
    # Print results
    print("\n" + "-"*60)
    print(f"{'Method':<15} {'Accuracy':<20} {'NMI':<20}")
    print("-"*60)
    for method, method_name in [('kmeans', 'K-means'), 
                                  ('urr', 'URR-SL'), 
                                  ('rurr', 'RURR-SL')]:
        acc_str = f"{stats[method]['acc_mean']:.4f} ± {stats[method]['acc_std']:.4f}"
        nmi_str = f"{stats[method]['nmi_mean']:.4f} ± {stats[method]['nmi_std']:.4f}"
        print(f"{method_name:<15} {acc_str:<20} {nmi_str:<20}")
    print("-"*60)
    
    return stats, results


def create_comparison_table(all_results):
    """Create a comparison table similar to Table I in paper"""
    
    # Prepare data for table
    data = []
    for dataset_name, stats in all_results.items():
        for method in ['kmeans', 'urr', 'rurr']:
            method_names = {'kmeans': 'K-means', 'urr': 'URR-SL', 'rurr': 'RURR-SL'}
            data.append({
                'Dataset': dataset_name,
                'Method': method_names[method],
                'Accuracy': f"{stats[method]['acc_mean']:.4f}",
                'Accuracy Std': f"{stats[method]['acc_std']:.4f}",
                'NMI': f"{stats[method]['nmi_mean']:.4f}",
                'NMI Std': f"{stats[method]['nmi_std']:.4f}"
            })
    
    df = pd.DataFrame(data)
    return df


def plot_method_comparison(all_results):
    """Create bar plots comparing methods across datasets"""
    
    datasets = list(all_results.keys())
    methods = ['kmeans', 'urr', 'rurr']
    method_names = ['K-means', 'URR-SL', 'RURR-SL']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        acc_means = [all_results[ds][method]['acc_mean'] for ds in datasets]
        acc_stds = [all_results[ds][method]['acc_std'] for ds in datasets]
        axes[0].bar(x + i*width, acc_means, width, 
                   label=name, yerr=acc_stds, capsize=5)
    
    axes[0].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Clustering Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(datasets, rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # NMI comparison
    for i, (method, name) in enumerate(zip(methods, method_names)):
        nmi_means = [all_results[ds][method]['nmi_mean'] for ds in datasets]
        nmi_stds = [all_results[ds][method]['nmi_std'] for ds in datasets]
        axes[1].bar(x + i*width, nmi_means, width, 
                   label=name, yerr=nmi_stds, capsize=5)
    
    axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('NMI', fontsize=12, fontweight='bold')
    axes[1].set_title('NMI Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(datasets, rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_improvement_heatmap(all_results):
    """Create heatmap showing improvement of RURR over baselines"""
    
    datasets = list(all_results.keys())
    comparisons = ['RURR vs K-means', 'RURR vs URR']
    metrics = ['Accuracy', 'NMI']
    
    # Calculate improvements
    improvements = np.zeros((len(datasets), len(comparisons) * len(metrics)))
    
    for i, ds in enumerate(datasets):
        rurr_acc = all_results[ds]['rurr']['acc_mean']
        rurr_nmi = all_results[ds]['rurr']['nmi_mean']
        kmeans_acc = all_results[ds]['kmeans']['acc_mean']
        kmeans_nmi = all_results[ds]['kmeans']['nmi_mean']
        urr_acc = all_results[ds]['urr']['acc_mean']
        urr_nmi = all_results[ds]['urr']['nmi_mean']
        
        improvements[i, 0] = (rurr_acc - kmeans_acc) * 100  # RURR vs K-means (Acc)
        improvements[i, 1] = (rurr_nmi - kmeans_nmi) * 100  # RURR vs K-means (NMI)
        improvements[i, 2] = (rurr_acc - urr_acc) * 100     # RURR vs URR (Acc)
        improvements[i, 3] = (rurr_nmi - urr_nmi) * 100     # RURR vs URR (NMI)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['RURR vs K-means\n(Acc %)', 'RURR vs K-means\n(NMI %)',
              'RURR vs URR\n(Acc %)', 'RURR vs URR\n(NMI %)']
    
    im = ax.imshow(improvements, cmap='RdYlGn', aspect='auto', vmin=-5, vmax=15)
    
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(datasets)
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(labels)):
            text = ax.text(j, i, f'{improvements[i, j]:.2f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Performance Improvement of RURR-SL (%)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.colorbar(im, ax=ax, label='Improvement (%)')
    plt.tight_layout()
    
    return fig


def main():
    """Main experiment runner"""
    
    print("="*70)
    print(" "*15 + "BENCHMARK EXPERIMENTS FOR RURR-SL")
    print("="*70)
    print("\nReproducing results from:")
    print("Zhang et al., 'Data Clustering via Uncorrelated Ridge Regression'")
    print("IEEE Trans. Neural Networks and Learning Systems, 2021\n")
    
    # Load datasets
    datasets = load_benchmark_datasets()
    
    # Run experiments on all datasets
    all_results = {}
    
    for ds_key, ds_info in datasets.items():
        stats, results = run_experiment(
            ds_info['X'], 
            ds_info['y'], 
            ds_info['n_clusters'],
            ds_info['name'],
            n_runs=10
        )
        all_results[ds_info['name']] = stats
    
    # Create comparison table
    print("\n" + "="*70)
    print("FINAL RESULTS TABLE (Similar to Table I in paper)")
    print("="*70)
    df = create_comparison_table(all_results)
    print(df.to_string(index=False))
    
    # Save table to CSV
    df.to_csv('benchmark_results.csv', index=False)
    print("\n✓ Results saved to: benchmark_results.csv")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    
    fig1 = plot_method_comparison(all_results)
    plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: method_comparison.png")
    
    fig2 = plot_improvement_heatmap(all_results)
    plt.savefig('improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: improvement_heatmap.png")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    avg_improvement_acc = 0
    avg_improvement_nmi = 0
    count = 0
    
    for ds in all_results:
        rurr_acc = all_results[ds]['rurr']['acc_mean']
        urr_acc = all_results[ds]['urr']['acc_mean']
        rurr_nmi = all_results[ds]['rurr']['nmi_mean']
        urr_nmi = all_results[ds]['urr']['nmi_mean']
        
        avg_improvement_acc += (rurr_acc - urr_acc) * 100
        avg_improvement_nmi += (rurr_nmi - urr_nmi) * 100
        count += 1
    
    avg_improvement_acc /= count
    avg_improvement_nmi /= count
    
    print(f"Average improvement of RURR-SL over URR-SL:")
    print(f"  • Accuracy: {avg_improvement_acc:+.2f}%")
    print(f"  • NMI: {avg_improvement_nmi:+.2f}%")
    print("\n✓ Experiments completed successfully!")
    print("="*70)
    
    plt.show()


if __name__ == "__main__":
    main()
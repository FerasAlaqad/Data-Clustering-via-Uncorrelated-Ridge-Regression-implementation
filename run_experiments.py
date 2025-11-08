"""
Comprehensive benchmark experiments for RURR-SL paper implementation.
Tests all algorithms on selected datasets and compares results.
"""

import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import algorithms
from rurr_implementation import RURR_SL, URR_SL, clustering_accuracy
from baseline_algorithms import KMeansClustering, RMKMC, FuzzyKMeans, RSFKM
from sklearn.metrics import normalized_mutual_info_score

# Import dataset loaders
from load_datasets import (
    load_glioma,
    load_att_faces,
    load_gt_faces,
    load_flower17,
)
from generate_toy_datasets import load_toy_dataset


# Available datasets
AVAILABLE_DATASETS = {
    "3cluster": {
        "name": "3-Cluster Gaussian",
        "loader": lambda: load_toy_dataset('3cluster'),
        "description": "Synthetic 3-cluster Gaussian data (Fig. 3)"
    },
    "multicluster": {
        "name": "Multicluster",
        "loader": lambda: load_toy_dataset('multicluster'),
        "description": "Synthetic multicluster data (Fig. 4)"
    },
    "glioma": {
        "name": "GLIOMA",
        "loader": lambda: load_glioma("datasets/GLIOMA.mat"),
        "description": "Gene expression data (50 samples, 4434 genes, 4 classes)"
    },
    "att_faces": {
        "name": "AT&T Faces",
        "loader": lambda: load_att_faces("datasets/att_faces"),
        "description": "Face recognition (400 images, 40 subjects)"
    },
    "gt_faces": {
        "name": "GT Faces",
        "loader": lambda: load_gt_faces("datasets/gt_db"),
        "description": "Georgia Tech faces (750 images, 50 subjects)"
    },
    "flower17": {
        "name": "FLOWER17",
        "loader": lambda: load_flower17("datasets/flower17"),
        "description": "Oxford flowers (1360 images, 17 categories)"
    },
}


def run_algorithm(algo_name: str, algo, X: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Run a single algorithm and measure performance.
    
    Parameters
    ----------
    algo_name : str
        Algorithm name for display
    algo : object
        Algorithm instance with fit() and predict() methods
    X : ndarray
        Data matrix (d, n)
    y_true : ndarray
        True labels
        
    Returns
    -------
    results : dict
        Dictionary containing accuracy, NMI, and runtime
    """
    print(f"    Running {algo_name}...", end=" ", flush=True)
    
    try:
        # Measure runtime
        start_time = time.time()
        algo.fit(X)
        y_pred = algo.predict()
        runtime = time.time() - start_time
        
        # Calculate metrics
        acc = clustering_accuracy(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        
        print(f"✓ (ACC: {acc:.4f}, NMI: {nmi:.4f}, Time: {runtime:.2f}s)")
        
        return {
            "accuracy": acc,
            "nmi": nmi,
            "runtime": runtime,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return {
            "accuracy": 0.0,
            "nmi": 0.0,
            "runtime": 0.0,
            "success": False,
            "error": str(e)
        }


def run_experiments_on_dataset(
    dataset_key: str,
    n_runs: int = 10,
    lambda_reg: float = 1.0,
    urr_alpha: float = 1.0
) -> Dict:
    """
    Run all algorithms on a single dataset multiple times.
    
    Parameters
    ----------
    dataset_key : str
        Dataset identifier
    n_runs : int
        Number of runs for averaging
    lambda_reg : float
        Regularization parameter for URR/RURR
    urr_alpha : float
        Fixed alpha for URR-SL
        
    Returns
    -------
    results : dict
        Results for all algorithms
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {AVAILABLE_DATASETS[dataset_key]['name']}")
    print(f"{'='*70}")
    
    # Load dataset
    print("  Loading dataset...", end=" ", flush=True)
    try:
        X, y_true = AVAILABLE_DATASETS[dataset_key]['loader']()
        n_clusters = len(np.unique(y_true))
        print(f"✓")
        print(f"    Shape: {X.shape} (features × samples)")
        print(f"    Clusters: {n_clusters}")
        print(f"    Samples per cluster: {np.bincount(y_true.astype(int))}")
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return None
    
    # Initialize algorithms
    algorithms = {
        "K-means": KMeansClustering(n_clusters=n_clusters, random_state=42),
        "RMKMC": RMKMC(n_clusters=n_clusters, random_state=42),
        "FKM": FuzzyKMeans(n_clusters=n_clusters, fuzziness=2.5, random_state=42),
        "RSFKM": RSFKM(n_clusters=n_clusters, fuzziness=2.0, cap=30, random_state=42),
        "URR-SL": URR_SL(n_clusters=n_clusters, alpha=urr_alpha, lambda_reg=lambda_reg, max_iter=100),
        "RURR-SL": RURR_SL(n_clusters=n_clusters, lambda_reg=lambda_reg, max_iter=100),
    }
    
    # Store results for each algorithm
    all_results = {name: [] for name in algorithms.keys()}
    
    # Run experiments
    print(f"\n  Running {n_runs} experiments:")
    for run_idx in range(n_runs):
        print(f"\n  Run {run_idx + 1}/{n_runs}:")
        
        for algo_name, algo in algorithms.items():
            result = run_algorithm(algo_name, algo, X, y_true)
            all_results[algo_name].append(result)
    
    # Compute statistics
    print(f"\n  Computing statistics...")
    stats = {}
    for algo_name, results in all_results.items():
        successful_runs = [r for r in results if r['success']]
        
        if successful_runs:
            stats[algo_name] = {
                "acc_mean": np.mean([r['accuracy'] for r in successful_runs]),
                "acc_std": np.std([r['accuracy'] for r in successful_runs]),
                "nmi_mean": np.mean([r['nmi'] for r in successful_runs]),
                "nmi_std": np.std([r['nmi'] for r in successful_runs]),
                "time_mean": np.mean([r['runtime'] for r in successful_runs]),
                "time_std": np.std([r['runtime'] for r in successful_runs]),
                "success_rate": len(successful_runs) / len(results),
            }
        else:
            stats[algo_name] = {
                "acc_mean": 0.0,
                "acc_std": 0.0,
                "nmi_mean": 0.0,
                "nmi_std": 0.0,
                "time_mean": 0.0,
                "time_std": 0.0,
                "success_rate": 0.0,
            }
    
    return stats


def print_results_table(all_results: Dict[str, Dict]):
    """
    Print formatted results table.
    
    Parameters
    ----------
    all_results : dict
        Results for all datasets and algorithms
    """
    print(f"\n{'='*100}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    for dataset_key, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n{AVAILABLE_DATASETS[dataset_key]['name']}:")
        print(f"{'-'*100}")
        print(f"{'Algorithm':<15} {'Accuracy':<20} {'NMI':<20} {'Time (s)':<20} {'Success':<10}")
        print(f"{'-'*100}")
        
        for algo_name, stats in results.items():
            acc_str = f"{stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}"
            nmi_str = f"{stats['nmi_mean']:.4f} ± {stats['nmi_std']:.4f}"
            time_str = f"{stats['time_mean']:.2f} ± {stats['time_std']:.2f}"
            success_str = f"{stats['success_rate']*100:.0f}%"
            
            print(f"{algo_name:<15} {acc_str:<20} {nmi_str:<20} {time_str:<20} {success_str:<10}")
    
    print(f"{'='*100}")


def save_results_to_file(all_results: Dict[str, Dict], output_file: str = "results.txt"):
    """
    Save results to a text file.
    
    Parameters
    ----------
    all_results : dict
        Results for all datasets and algorithms
    output_file : str
        Output filename
    """
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("RURR-SL BENCHMARK RESULTS\n")
        f.write("="*100 + "\n\n")
        
        for dataset_key, results in all_results.items():
            if results is None:
                continue
            
            f.write(f"\n{AVAILABLE_DATASETS[dataset_key]['name']}:\n")
            f.write(f"{AVAILABLE_DATASETS[dataset_key]['description']}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Algorithm':<15} {'Accuracy':<20} {'NMI':<20} {'Time (s)':<20} {'Success':<10}\n")
            f.write("-"*100 + "\n")
            
            for algo_name, stats in results.items():
                acc_str = f"{stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}"
                nmi_str = f"{stats['nmi_mean']:.4f} ± {stats['nmi_std']:.4f}"
                time_str = f"{stats['time_mean']:.2f} ± {stats['time_std']:.2f}"
                success_str = f"{stats['success_rate']*100:.0f}%"
                
                f.write(f"{algo_name:<15} {acc_str:<20} {nmi_str:<20} {time_str:<20} {success_str:<10}\n")
            
            f.write("\n")
    
    print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run clustering experiments on selected datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  3cluster      - 3-Cluster Gaussian (synthetic)
  multicluster  - Multicluster (synthetic)
  glioma        - GLIOMA gene expression
  att_faces     - AT&T face database
  gt_faces      - Georgia Tech faces
  flower17      - Oxford 17 flowers

Examples:
  # Run on all datasets
  python run_experiments.py --all
  
  # Run on specific datasets
  python run_experiments.py --datasets 3cluster glioma att_faces
  
  # Run with custom parameters
  python run_experiments.py --datasets 3cluster --runs 5 --lambda 10.0
        """
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(AVAILABLE_DATASETS.keys()),
        help='Datasets to run experiments on'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run on all available datasets'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=10,
        help='Number of runs per dataset (default: 10)'
    )
    parser.add_argument(
        '--lambda',
        type=float,
        default=1.0,
        dest='lambda_reg',
        help='Regularization parameter for URR/RURR (default: 1.0)'
    )
    parser.add_argument(
        '--urr-alpha',
        type=float,
        default=1.0,
        help='Fixed alpha for URR-SL (default: 1.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.txt',
        help='Output file for results (default: results.txt)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list:
        print("\nAvailable datasets:")
        print("="*70)
        for key, info in AVAILABLE_DATASETS.items():
            print(f"  {key:<15} - {info['name']}")
            print(f"  {'':15}   {info['description']}")
        print("="*70)
        return
    
    # Determine which datasets to run
    if args.all:
        datasets_to_run = list(AVAILABLE_DATASETS.keys())
    elif args.datasets:
        datasets_to_run = args.datasets
    else:
        # Interactive mode
        print("\nAvailable datasets:")
        for i, (key, info) in enumerate(AVAILABLE_DATASETS.items(), 1):
            print(f"  {i}. {key:<15} - {info['name']}")
        
        print("\nEnter dataset numbers (space-separated) or 'all': ", end="")
        user_input = input().strip()
        
        if user_input.lower() == 'all':
            datasets_to_run = list(AVAILABLE_DATASETS.keys())
        else:
            try:
                indices = [int(x) - 1 for x in user_input.split()]
                dataset_keys = list(AVAILABLE_DATASETS.keys())
                datasets_to_run = [dataset_keys[i] for i in indices]
            except (ValueError, IndexError):
                print("Invalid input. Exiting.")
                return
    
    # Print experiment configuration
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Runs per dataset: {args.runs}")
    print(f"Lambda (regularization): {args.lambda_reg}")
    print(f"URR-SL alpha: {args.urr_alpha}")
    print("="*70)
    
    # Run experiments
    all_results = {}
    for dataset_key in datasets_to_run:
        results = run_experiments_on_dataset(
            dataset_key,
            n_runs=args.runs,
            lambda_reg=args.lambda_reg,
            urr_alpha=args.urr_alpha
        )
        all_results[dataset_key] = results
    
    # Print and save results
    print_results_table(all_results)
    save_results_to_file(all_results, args.output)
    
    print(f"\n✓ Experiments completed!")


if __name__ == "__main__":
    main()


"""
Comprehensive benchmark experiments exactly following the paper's methodology.

Paper Section VI.C states:
- Each experiment run 10 times
- λ searched in [10^-4, 10^-3, 10^-2, 10^-1, 1, 10^1, 10^2, 10^3, 10^4]
- FKM fuzzy level = 2.5
- RSFKM parameter tuned in [10, 15, 20, 25, 30, 35, 40, 45, 50]
- Real cluster number c is pre-given
"""

import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import json
warnings.filterwarnings('ignore')

# Import algorithms
from rurr_implementation import RURR_SL, URR_SL, clustering_accuracy, normalized_mutual_info

# Import baseline algorithms
from baseline_algorithms import KMeansClustering, RMKMC, FuzzyKMeans, RSFKM

# Import dataset loaders
from load_datasets import (
    load_glioma,
    load_colon_csv,
    load_att_faces,
    load_gt_faces,
    load_flower17,
    load_imm_faces,
    load_toy_dataset,
)


# Paper Section VI.C: λ parameter search grid
LAMBDA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]

# Paper Section VI.C: RSFKM parameter search grid
RSFKM_CAP_GRID = [10, 15, 20, 25, 30, 35, 40, 45, 50]


# Available datasets (Paper Table I + Additional Tests)
AVAILABLE_DATASETS = {
    # Paper Synthetic Datasets (Figures 3-4)
    "3cluster": {
        "name": "3-Cluster Gaussian",
        "loader": lambda: load_toy_dataset('3cluster', use_saved=True, allow_generate=False),
        "description": "Synthetic 3-cluster Gaussian data (Paper Fig. 3)"
    },
    "multicluster": {
        "name": "Multicluster",
        "loader": lambda: load_toy_dataset('multicluster', use_saved=True, allow_generate=False),
        "description": "Synthetic multicluster data (Paper Fig. 4)"
    },
    # Additional Synthetic Datasets
    "nested_circles": {
        "name": "Nested Circles",
        "loader": lambda: load_toy_dataset('nested_circles', use_saved=True, allow_generate=False),
        "description": "Nested circles clustering problem"
    },
    "moons": {
        "name": "Two Moons",
        "loader": lambda: load_toy_dataset('moons', use_saved=True, allow_generate=False),
        "description": "Two moons clustering problem"
    },
    # Paper Real Datasets (Table I)
    "colon": {
        "name": "COLON",
        "loader": lambda: load_colon_csv("datasets/colon - labled.csv"),
        "description": "Colon cancer gene expression (62 samples, 2000 genes, 2 classes)"
    },
    "glioma": {
        "name": "GLIOMA",
        "loader": lambda: load_glioma("datasets/GLIOMA.mat"),
        "description": "Gene expression data (50 samples, 4434 genes, 4 classes)"
    },
    "att_faces": {
        "name": "AT&T",
        "loader": lambda: load_att_faces("datasets/att_faces"),
        "description": "AT&T face database (400 images, 40 subjects)"
    },
    "gt_faces": {
        "name": "GT",
        "loader": lambda: load_gt_faces("datasets/gt_db"),
        "description": "Georgia Tech faces (750 images, 50 subjects)"
    },
    "flower17": {
        "name": "FLOWER17",
        "loader": lambda: load_flower17("datasets/flower17"),
        "description": "Oxford flowers (1360 images, 17 categories)"
    },
    "imm": {
        "name": "IMM",
        "loader": lambda: load_imm_faces("datasets/imm"),
        "description": "IMM face database (240 images, 40 subjects)"
    },
}


PRESET_PARAMS = {
    "3cluster": {"lambda_rurr": 1e-4, "lambda_urr": 1e-4, "rsfkm_cap": 10},
    "multicluster": {"lambda_rurr": 1e-4, "lambda_urr": 1e4, "rsfkm_cap": 40},
    "nested_circles": {"lambda_rurr": 1e-1, "lambda_urr": 1e2, "rsfkm_cap": 10},
    "moons": {"lambda_rurr": 1e-4, "lambda_urr": 1.0, "rsfkm_cap": 10},
    "colon": {"lambda_rurr": 1e2, "lambda_urr": 1e2, "rsfkm_cap": 10},
    "att_faces": {"lambda_rurr": 1e2, "lambda_urr": 1e2, "rsfkm_cap": 30},
    "gt_faces": {"lambda_rurr": 1e2, "lambda_urr": 1e2, "rsfkm_cap": 30},
    "flower17": {"lambda_rurr": 1e2, "lambda_urr": 1e2, "rsfkm_cap": 30},
}


def prompt_user_for_datasets(default_selection: List[str]) -> List[str]:
    """
    Prompt the user to choose which datasets to train/test on.
    
    Parameters
    ----------
    default_selection : list of str
        Dataset keys used when the user presses Enter without a choice.
    
    Returns
    -------
    list of str
        Valid dataset keys selected by the user.
    """
    print("\nAvailable datasets:")
    indexed_keys = list(AVAILABLE_DATASETS.keys())
    for idx, key in enumerate(indexed_keys, start=1):
        meta = AVAILABLE_DATASETS[key]
        print(f"  {idx:>2}. {key:<15} ({meta['name']})")
    
    default_display = ", ".join(default_selection)
    prompt = (
        "\nEnter dataset numbers separated by space or comma (e.g., '1 4 7'),\n"
        "type 'all' to run every dataset, or press Enter to use the default selection.\n"
        f"Default selection: [{default_display}]\n"
        "Selection: "
    )
    
    while True:
        user_input = input(prompt).strip()
        
        if not user_input:
            return default_selection
        
        if user_input.lower() == 'all':
            return list(AVAILABLE_DATASETS.keys())
        
        tokens = [token.strip() for token in user_input.replace(",", " ").split()]
        selections = []
        invalid = []
        
        for token in tokens:
            if not token:
                continue
            if not token.isdigit():
                invalid.append(token)
                continue
            index = int(token)
            if index < 1 or index > len(indexed_keys):
                invalid.append(token)
                continue
            selections.append(indexed_keys[index - 1])
        
        if invalid:
            print(f"\nInvalid dataset key(s): {', '.join(invalid)}")
            print("Please enter numbers shown in the dataset list.")
            continue
        
        return selections


def tune_lambda_parameter(X: np.ndarray, y_true: np.ndarray, n_clusters: int, 
                          algorithm_class, n_trials: int = 3) -> float:
    """
    Tune λ parameter using grid search as specified in Paper Section VI.C
    
    Parameters
    ----------
    X : ndarray of shape (d, n)
        Data matrix
    y_true : ndarray
        True labels
    n_clusters : int
        Number of clusters
    algorithm_class : class
        Either RURR_SL or URR_SL
    n_trials : int
        Number of trials per λ value (for averaging)
        
    Returns
    -------
    best_lambda : float
        Best λ value
    """
    best_lambda = 1.0
    best_acc = 0.0
    
    for lambda_val in LAMBDA_GRID:
        acc_list = []
        
        for trial in range(n_trials):
            seed = 42 + trial
            
            if algorithm_class == RURR_SL:
                algo = RURR_SL(n_clusters=n_clusters, lambda_reg=lambda_val, 
                              max_iter=100, random_state=seed)
            else:  # URR_SL
                algo = URR_SL(n_clusters=n_clusters, alpha=1.0, lambda_reg=lambda_val,
                             max_iter=100, random_state=seed)
            
            try:
                algo.fit(X)
                y_pred = algo.predict()
                acc = clustering_accuracy(y_true, y_pred)
                acc_list.append(acc)
            except:
                pass
        
        if acc_list:
            avg_acc = np.mean(acc_list)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lambda = lambda_val
    
    return best_lambda


def tune_rsfkm_parameter(X: np.ndarray, y_true: np.ndarray, n_clusters: int,
                         n_trials: int = 3) -> int:
    """
    Tune RSFKM cap parameter as specified in Paper Section VI.C
    
    Parameters
    ----------
    X : ndarray of shape (d, n)
        Data matrix
    y_true : ndarray
        True labels
    n_clusters : int
        Number of clusters
    n_trials : int
        Number of trials per cap value
        
    Returns
    -------
    best_cap : int
        Best cap value
    """
    best_cap = 30
    best_acc = 0.0
    
    for cap_val in RSFKM_CAP_GRID:
        acc_list = []
        
        for trial in range(n_trials):
            seed = 42 + trial
            algo = RSFKM(n_clusters=n_clusters, fuzziness=2.0, cap=cap_val, 
                        max_iter=100, random_state=seed)
            
            try:
                algo.fit(X)
                y_pred = algo.predict()
                acc = clustering_accuracy(y_true, y_pred)
                acc_list.append(acc)
            except:
                pass
        
        if acc_list:
            avg_acc = np.mean(acc_list)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_cap = cap_val
    
    return best_cap


def run_single_experiment(algo_name: str, algo, X: np.ndarray, y_true: np.ndarray) -> Dict:
    """
    Run a single algorithm and measure performance
    
    Parameters
    ----------
    algo_name : str
        Algorithm name
    algo : object
        Algorithm instance
    X : ndarray of shape (d, n)
        Data matrix
    y_true : ndarray
        True labels
        
    Returns
    -------
    results : dict
        Performance metrics
    """
    try:
        start_time = time.time()
        algo.fit(X)
        y_pred = algo.predict()
        runtime = time.time() - start_time
        
        acc = clustering_accuracy(y_true, y_pred)
        nmi = normalized_mutual_info(y_true, y_pred)
        
        return {
            "accuracy": acc,
            "nmi": nmi,
            "runtime": runtime,
            "success": True,
            "error": None
        }
    
    except Exception as e:
        return {
            "accuracy": 0.0,
            "nmi": 0.0,
            "runtime": 0.0,
            "success": False,
            "error": str(e)
        }


def run_experiments_on_dataset(dataset_key: str, n_runs: int = 10, 
                               tune_params: bool = True) -> Dict:
    """
    Run all algorithms on a single dataset following paper's methodology
    
    Paper Section VI.C:
    - Run each experiment 10 times
    - Tune λ for URR-SL and RURR-SL
    - Tune cap for RSFKM
    - FKM uses fuzzy level = 2.5
    
    Parameters
    ----------
    dataset_key : str
        Dataset identifier
    n_runs : int
        Number of runs (paper uses 10)
    tune_params : bool
        Whether to tune parameters (paper does this)
        
    Returns
    -------
    results : dict
        Results for all algorithms
    """
    print(f"\n{'='*80}")
    print(f"Dataset: {AVAILABLE_DATASETS[dataset_key]['name']}")
    print(f"{'='*80}")
    
    # Load dataset
    print("  Loading dataset...", end=" ", flush=True)
    try:
        X, y_true = AVAILABLE_DATASETS[dataset_key]['loader']()
        n_clusters = len(np.unique(y_true))
        print(f"✓")
        print(f"    Shape: {X.shape} (features × samples)")
        print(f"    Clusters: {n_clusters}")
        print(f"    Samples: {np.bincount(y_true.astype(int))}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None
    
    # Tune parameters (Paper Section VI.C)
    if tune_params:
        print("\n  Tuning parameters...")
        
        print("    Tuning λ for RURR-SL...", end=" ", flush=True)
        best_lambda_rurr = tune_lambda_parameter(X, y_true, n_clusters, RURR_SL, n_trials=3)
        print(f"✓ Best λ = {best_lambda_rurr}")
        
        print("    Tuning λ for URR-SL...", end=" ", flush=True)
        best_lambda_urr = tune_lambda_parameter(X, y_true, n_clusters, URR_SL, n_trials=3)
        print(f"✓ Best λ = {best_lambda_urr}")
        
        print("    Tuning cap for RSFKM...", end=" ", flush=True)
        best_cap = tune_rsfkm_parameter(X, y_true, n_clusters, n_trials=3)
        print(f"✓ Best cap = {best_cap}")
    else:
        preset = PRESET_PARAMS.get(dataset_key, {})
        best_lambda_rurr = preset.get("lambda_rurr", 1.0)
        best_lambda_urr = preset.get("lambda_urr", 1.0)
        best_cap = preset.get("rsfkm_cap", 30)
        print("\n  Using preset parameters:")
        print(f"    RURR-SL λ = {best_lambda_rurr}")
        print(f"    URR-SL  λ = {best_lambda_urr}")
        print(f"    RSFKM cap = {best_cap}")
    
    # Run experiments (Paper: 10 times)
    print(f"\n  Running {n_runs} experiments:")
    
    all_results = {
        "K-means": [],
        "RMKMC": [],
        "FKM": [],
        "RSFKM": [],
        "URR-SL": [],
        "RURR-SL": []
    }
    
    for run_idx in range(n_runs):
        print(f"\n    Run {run_idx + 1}/{n_runs}:")
        seed = 42 + run_idx
        
        # Paper Section VI.C: "k-means and RMKMC are parameter-free"
        algorithms = {
            "K-means": KMeansClustering(n_clusters=n_clusters, random_state=seed),
            "RMKMC": RMKMC(n_clusters=n_clusters, random_state=seed),
            # Paper Section VI.C: "fuzzy level is chosen as 2.5"
            "FKM": FuzzyKMeans(n_clusters=n_clusters, fuzziness=2.5, random_state=seed),
            # Paper Section VI.C: tuned cap parameter
            "RSFKM": RSFKM(n_clusters=n_clusters, fuzziness=2.0, cap=best_cap, random_state=seed),
            # Paper: URR-SL with α=1 and tuned λ
            "URR-SL": URR_SL(n_clusters=n_clusters, alpha=1.0, lambda_reg=best_lambda_urr, 
                            max_iter=100, random_state=seed),
            # Paper: RURR-SL with tuned λ
            "RURR-SL": RURR_SL(n_clusters=n_clusters, lambda_reg=best_lambda_rurr,
                              max_iter=100, random_state=seed),
        }
        
        for algo_name, algo in algorithms.items():
            print(f"      {algo_name:12s} ...", end=" ", flush=True)
            result = run_single_experiment(algo_name, algo, X, y_true)
            all_results[algo_name].append(result)
            
            if result['success']:
                print(f"✓ ACC={result['accuracy']:.4f} NMI={result['nmi']:.4f}")
            else:
                print(f"✗ Failed")
    
    # Compute statistics (Paper Table I reports mean ± std)
    print("\n  Computing statistics...")
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
                "n_success": len(successful_runs)
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
                "n_success": 0
            }
    
    return stats


def print_results_table(all_results: Dict[str, Dict]):
    """
    Print results table in paper format (Table I style)
    """
    print(f"\n{'='*100}")
    print("FINAL RESULTS (Paper Table I Format)")
    print(f"{'='*100}")
    
    for dataset_key, results in all_results.items():
        if results is None:
            continue
        
        print(f"\n{AVAILABLE_DATASETS[dataset_key]['name']}:")
        print(f"{'-'*100}")
        print(f"{'Algorithm':<12} {'Accuracy':<25} {'NMI':<25} {'Time (s)':<20}")
        print(f"{'-'*100}")
        
        for algo_name in ["K-means", "RMKMC", "FKM", "RSFKM", "URR-SL", "RURR-SL"]:
            if algo_name not in results:
                continue
            
            stats = results[algo_name]
            
            # Format as mean ± std (Paper Table I format)
            acc_str = f"{stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}"
            nmi_str = f"{stats['nmi_mean']:.4f} ± {stats['nmi_std']:.4f}"
            time_str = f"{stats['time_mean']:.2f} ± {stats['time_std']:.2f}"
            
            print(f"{algo_name:<12} {acc_str:<25} {nmi_str:<25} {time_str:<20}")
        
        print()
    
    print(f"{'='*100}")


def save_results_to_file(all_results: Dict[str, Dict], output_file: str = "paper_results.txt"):
    """
    Save results to file in paper format
    """
    with open(output_file, 'w') as f:
        f.write("="*100 + "\n")
        f.write("RURR-SL BENCHMARK RESULTS (Paper Methodology)\n")
        f.write("="*100 + "\n\n")
        
        for dataset_key, results in all_results.items():
            if results is None:
                continue
            
            f.write(f"\n{AVAILABLE_DATASETS[dataset_key]['name']}:\n")
            f.write(f"{AVAILABLE_DATASETS[dataset_key]['description']}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Algorithm':<12} {'Accuracy':<25} {'NMI':<25} {'Time (s)':<20}\n")
            f.write("-"*100 + "\n")
            
            for algo_name in ["K-means", "RMKMC", "FKM", "RSFKM", "URR-SL", "RURR-SL"]:
                if algo_name not in results:
                    continue
                
                stats = results[algo_name]
                acc_str = f"{stats['acc_mean']:.4f} ± {stats['acc_std']:.4f}"
                nmi_str = f"{stats['nmi_mean']:.4f} ± {stats['nmi_std']:.4f}"
                time_str = f"{stats['time_mean']:.2f} ± {stats['time_std']:.2f}"
                
                f.write(f"{algo_name:<12} {acc_str:<25} {nmi_str:<25} {time_str:<20}\n")
            
            f.write("\n")
    
    print(f"\n✓ Results saved to {output_file}")


def save_results_to_json(all_results: Dict[str, Dict], output_file: str = "paper_results.json"):
    """
    Save raw numerical results to a JSON file for downstream processing (e.g., plotting)
    """

    def _to_builtin(value):
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    serializable = {}

    for dataset_key, results in all_results.items():
        if results is None:
            continue

        dataset_meta = AVAILABLE_DATASETS.get(dataset_key, {})
        serializable[dataset_key] = {
            "name": dataset_meta.get("name", dataset_key),
            "description": dataset_meta.get("description", ""),
            "metrics": {}
        }

        for algo_name, stats in results.items():
            serializable[dataset_key]["metrics"][algo_name] = {
                metric_key: _to_builtin(metric_value)
                for metric_key, metric_value in stats.items()
            }

    with open(output_file, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"✓ Results (JSON) saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run clustering experiments following exact paper methodology",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--datasets', nargs='+', choices=list(AVAILABLE_DATASETS.keys()),
                       help='Datasets to run')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs (paper uses 10)')
    parser.add_argument('--tune', action='store_true', help='Run parameter tuning instead of using presets')
    parser.add_argument('--output', type=str, default='paper_results.txt', help='Output file')
    parser.add_argument('--json-output', type=str, default='paper_results.json',
                       help='Path to save raw numerical results in JSON format')
    
    args = parser.parse_args()
    
    # Determine datasets (prompt user before starting)
    if args.all:
        initial_selection = list(AVAILABLE_DATASETS.keys())
    elif args.datasets:
        initial_selection = args.datasets
    else:
        initial_selection = ['3cluster', 'glioma', 'att_faces']  # Default subset
    
    datasets_to_run = prompt_user_for_datasets(initial_selection)
    
    # Print configuration
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION (Following Paper Methodology)")
    print("="*80)
    print(f"Datasets: {', '.join(datasets_to_run)}")
    print(f"Runs per dataset: {args.runs} (Paper uses 10)")
    print(f"Parameter tuning: {'Yes (grid search)' if args.tune else 'No (using preset values)'}")
    print(f"λ grid: {LAMBDA_GRID}")
    print(f"RSFKM cap grid: {RSFKM_CAP_GRID}")
    print("="*80)
    
    # Run experiments
    all_results = {}
    for dataset_key in datasets_to_run:
        results = run_experiments_on_dataset(
            dataset_key,
            n_runs=args.runs,
            tune_params=args.tune
        )
        all_results[dataset_key] = results
    
    # Print and save results
    print_results_table(all_results)
    save_results_to_file(all_results, args.output)
    if args.json_output:
        save_results_to_json(all_results, args.json_output)
    


if __name__ == "__main__":
    main()
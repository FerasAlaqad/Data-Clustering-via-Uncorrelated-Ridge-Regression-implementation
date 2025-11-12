"""
Generate publication-ready figures from clustering experiment results.

This script consumes the JSON artifact produced by `run_experiments.py`
and creates both per-dataset and cross-dataset figures that summarise
the mean (± std) Accuracy, NMI, and Runtime for each algorithm.
All statistics correspond to the average over the paper's 10 runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


# Algorithm display order (matching paper Table I)
ALGORITHM_ORDER: Sequence[str] = [
    "K-means",
    "RMKMC",
    "FKM",
    "RSFKM",
    "URR-SL",
    "RURR-SL",
]

# Metric specification: (mean_key, std_key, display_name)
METRICS = [
    ("acc_mean", "acc_std", "Accuracy"),
    ("nmi_mean", "nmi_std", "Normalized Mutual Information"),
    ("time_mean", "time_std", "Runtime (s)"),
]

# Consistent colour palette across figures
ALGORITHM_COLOURS = {
    "K-means": "#4C72B0",
    "RMKMC": "#55A868",
    "FKM": "#C44E52",
    "RSFKM": "#8172B3",
    "URR-SL": "#CCB974",
    "RURR-SL": "#64B5CD",
}

FIGURE_CAPTIONS = {
    "3cluster": "Figure 1. Accuracy, NMI, and runtime for the 3-Cluster Gaussian dataset.",
    "multicluster": "Figure 2. Accuracy, NMI, and runtime for the Multicluster dataset.",
    "glioma": "Figure 3. Accuracy, NMI, and runtime for the GLIOMA dataset.",
    "moons": "Figure 4. Accuracy, NMI, and runtime for the Two Moons dataset.",
    "nested_circles": "Figure 5. Accuracy, NMI, and runtime for the Nested Circles dataset.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate figures from clustering benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper_results.json"),
        help="Path to the JSON results file produced by run_experiments.py",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to plot (defaults to all datasets present in the JSON file)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where the generated figures will be saved",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        help="Image formats to export (e.g., png pdf svg)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Render resolution (dots per inch) for raster outputs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip per-dataset figures and only generate cross-dataset comparisons",
    )
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Results file '{path}' not found. Run run_experiments.py first."
        )
    with path.open("r") as f:
        return json.load(f)


def prepare_series(dataset_metrics: Dict[str, Dict], mean_key: str, std_key: str):
    means = []
    stds = []
    for algo in ALGORITHM_ORDER:
        algo_stats = dataset_metrics.get(algo, {})
        means.append(algo_stats.get(mean_key, np.nan))
        stds.append(algo_stats.get(std_key, np.nan))
    return np.array(means, dtype=float), np.array(stds, dtype=float)


def plot_dataset(
    dataset_key: str,
    dataset_payload: Dict,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    metrics = dataset_payload.get("metrics", {})
    dataset_name = dataset_payload.get("name", dataset_key)
    exported_paths: List[Path] = []

    try:
        plt.style.use("seaborn-v0_8")
    except (OSError, ValueError):
        # Style might be unavailable depending on the Matplotlib version.
        plt.style.use("default")

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.9))

    x = np.arange(len(ALGORITHM_ORDER))
    width = 0.65

    for ax, (mean_key, std_key, display_name) in zip(axes, METRICS):
        means, stds = prepare_series(metrics, mean_key, std_key)

        bars = ax.bar(
            x,
            means,
            width=width,
            color=[ALGORITHM_COLOURS.get(algo, "#333333") for algo in ALGORITHM_ORDER],
            edgecolor="black",
            linewidth=0.6,
            yerr=stds,
            capsize=4,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHM_ORDER, rotation=35, ha="right")
        ax.set_ylabel(display_name)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

        # Emphasise the proposed method with a thicker outline
        for bar, algo in zip(bars, ALGORITHM_ORDER):
            if algo == "RURR-SL":
                bar.set_edgecolor("black")
                bar.set_linewidth(1.5)

        # Adjust y-limit to avoid clipping error bars when std is available
        finite_heights = means[np.isfinite(means)]
        finite_stds = stds[np.isfinite(stds)]
        if finite_heights.size > 0:
            if display_name.lower().startswith("accuracy"):
                bottom = max(0.0, np.min(np.minimum(finite_heights - finite_stds, finite_heights)))
                ax.set_ylim(0.0, 1.0)
            else:
                top = np.max(finite_heights + finite_stds) * 1.15
                bottom = np.min(np.minimum(finite_heights - finite_stds, finite_heights))
                ax.set_ylim(bottom * 0.98 if bottom >= 0 else bottom * 1.1, top)

    fig.suptitle(dataset_name, fontsize=14, fontweight="bold")
    caption = FIGURE_CAPTIONS.get(dataset_key)
    if caption:
        fig.text(0.5, 0.02, caption, ha="center", va="center", fontsize=11)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        target_path = output_dir / f"{dataset_key}_metrics.{fmt}"
        fig.savefig(target_path, dpi=dpi, bbox_inches="tight")
        exported_paths.append(target_path)

    plt.close(fig)
    return exported_paths


def plot_cross_dataset(
    dataset_keys: Sequence[str],
    results: Dict[str, Dict],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    exported_paths: List[Path] = []

    dataset_names = [
        results[key].get("name", key) for key in dataset_keys
    ]

    x = np.arange(len(dataset_keys))
    n_algorithms = len(ALGORITHM_ORDER)
    width = 0.8 / max(n_algorithms, 1)

    try:
        plt.style.use("seaborn-v0_8")
    except (OSError, ValueError):
        plt.style.use("default")

    for mean_key, std_key, display_name in METRICS:
        fig, ax = plt.subplots(figsize=(5.5, 4.8))

        for idx, algo in enumerate(ALGORITHM_ORDER):
            means = []
            stds = []

            for dataset_key in dataset_keys:
                dataset_metrics = results[dataset_key].get("metrics", {})
                algo_stats = dataset_metrics.get(algo, {})
                means.append(algo_stats.get(mean_key, np.nan))
                stds.append(algo_stats.get(std_key, np.nan))

            means = np.array(means, dtype=float)
            stds = np.array(stds, dtype=float)

            offsets = x + (idx - (n_algorithms - 1) / 2) * width

            bar_container = ax.bar(
                offsets,
                means,
                width=width * 0.92,
                yerr=stds,
                capsize=4,
                label=algo,
                color=ALGORITHM_COLOURS.get(algo, "#333333"),
                edgecolor="black",
                linewidth=0.6,
            )

            if algo == "RURR-SL":
                for bar in bar_container:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=20, ha="right")
        ax.set_ylabel(display_name)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)

        # Avoid clipping error bars
        bars_heights = []
        bars_stds = []
        for dataset_key in dataset_keys:
            dataset_metrics = results[dataset_key].get("metrics", {})
            for algo in ALGORITHM_ORDER:
                algo_stats = dataset_metrics.get(algo, {})
                bars_heights.append(algo_stats.get(mean_key, np.nan))
                bars_stds.append(algo_stats.get(std_key, np.nan))

        finite_heights = np.array(bars_heights, dtype=float)[
            np.isfinite(bars_heights)
        ]
        finite_stds = np.array(bars_stds, dtype=float)[
            np.isfinite(bars_stds)
        ]

        if finite_heights.size > 0:
            if display_name.lower().startswith("accuracy"):
                ax.set_ylim(0.0, 1.0)
            else:
                top = np.max(finite_heights + finite_stds) * 1.15
                bottom = np.min(np.minimum(finite_heights - finite_stds, finite_heights))
                ax.set_ylim(bottom * 0.98 if bottom >= 0 else bottom * 1.1, top)

        fig.tight_layout(rect=[0, 0, 1, 0.93])

        output_dir.mkdir(parents=True, exist_ok=True)
        for fmt in formats:
            fmt = fmt.lower().lstrip(".")
            target_path = output_dir / f"comparison_{mean_key}.{fmt}"
            fig.savefig(target_path, dpi=dpi, bbox_inches="tight")
            exported_paths.append(target_path)

        plt.close(fig)

    return exported_paths


def main():
    args = parse_args()
    results = load_results(args.input)

    datasets_to_plot = args.datasets or list(results.keys())

    missing = [ds for ds in datasets_to_plot if ds not in results]
    if missing:
        raise KeyError(
            f"Dataset(s) {missing} not present in '{args.input}'. "
            "Verify the results file or run the experiments again."
        )

    all_exports: List[Path] = []

    if not args.skip_individual:
        for dataset_key in datasets_to_plot:
            exports = plot_dataset(
                dataset_key,
                results[dataset_key],
                args.output_dir,
                args.formats,
                args.dpi,
            )
            all_exports.extend(exports)

    if len(datasets_to_plot) > 1:
        comparison_exports = plot_cross_dataset(
            datasets_to_plot,
            results,
            args.output_dir,
            args.formats,
            args.dpi,
        )
        all_exports.extend(comparison_exports)

    print("Generated figures:")
    for path in all_exports:
        print(f"  - {path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()


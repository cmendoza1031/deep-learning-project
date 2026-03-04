"""
Generates the key result tables and visualizations.
Run AFTER all 4 experiments complete.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC']
CLASS_RISK = {0: 'HIGH', 1: 'MED', 2: 'MED', 3: 'LOW', 4: 'LOW', 5: 'LOW', 6: 'LOW'}
METHODS = ['fedavg', 'hetlora', 'flexlora', 'dqaw']
METHOD_LABELS = {
    'fedavg': 'FedAvg (Homogeneous r=8)',
    'hetlora': 'HetLoRA (Zero-Pad)',
    'flexlora': 'FlexLoRA',
    'dqaw': 'DQAW (Ours)',
}


def load_results(results_dir: str = './results'):
    results = {}
    for method in METHODS:
        path = Path(results_dir) / method / f'{method}_results.json'
        if path.exists():
            with open(path) as f:
                results[method] = json.load(f)
    return results


def plot_bacc_over_rounds(results: dict, save_path: str = './results/bacc_curves.png'):
    """Plot average BACC over rounds for all methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'fedavg': 'gray', 'hetlora': 'red', 'flexlora': 'blue', 'dqaw': 'green'}
    linestyles = {'fedavg': '--', 'hetlora': '-', 'flexlora': '-', 'dqaw': '-'}

    for method, data in results.items():
        baccs = data['bacc_per_client']
        avg_baccs = [sum(r) / len(r) for r in baccs]
        rounds = data['rounds']

        ax.plot(rounds, avg_baccs,
                label=METHOD_LABELS[method],
                color=colors.get(method, 'black'),
                linestyle=linestyles.get(method, '-'),
                linewidth=2)

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Average BACC', fontsize=12)
    ax.set_title('Federated Training: BACC Over Rounds\n(FedISIC, 4 Hospitals)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def plot_final_per_class_recall(
    results: dict,
    save_path: str = './results/per_class_recall.png'
):
    """
    Per-class recall comparison at final round.
    Shows that rank collapse (HetLoRA) specifically harms MEL recall.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (class_id, class_name) in zip(axes, [(0, 'MEL (Melanoma)'), (1, 'NV (Nevus)')]):
        method_names = [METHOD_LABELS[m] for m in results]
        method_recalls = _extract_per_class_recall(results, class_id)

        colors = ['gray', 'red', 'blue', 'green']
        ax.bar(range(len(method_names)), method_recalls, color=colors[:len(method_names)])
        ax.set_xticks(range(len(method_names)))
        ax.set_xticklabels(method_names, rotation=15, ha='right')
        ax.set_ylabel('Recall', fontsize=12)
        ax.set_title(f'{class_name} Recall\n(Clinical Importance: {"HIGH" if class_id == 0 else "MED"})',
                     fontsize=12)
        ax.set_ylim(0, 1)

    plt.suptitle('Class-Specific Recall: Rank Collapse Harms High-Stakes Classes',
                 fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")


def _extract_per_class_recall(results: dict, class_id: int):
    """Extract per-class recall for a given class from results."""
    recalls = []
    for method, data in results.items():
        if 'per_class_recall' in data and data['per_class_recall']:
            final_round = data['per_class_recall'][-1]
            # final_round is list of per-client dicts: [{0: recall, 1: recall, ...}, ...]
            client_recalls = [pc.get(class_id, 0) for pc in final_round if isinstance(pc, dict)]
            recall = sum(client_recalls) / len(client_recalls) if client_recalls else 0
        else:
            recall = 0
        recalls.append(recall)
    return recalls


def print_results_table(results: dict):
    """Prints the main results table for paper/presentation."""
    print("\n" + "="*70)
    print("RESULTS TABLE: Final Round Performance")
    print("="*70)
    print(f"{'Method':<30} {'Avg BACC':>10} {'MEL Recall':>12} {'NV Recall':>10}")
    print("-"*70)

    for method, data in results.items():
        final_baccs = data['bacc_per_client'][-1]
        avg_bacc = sum(final_baccs) / len(final_baccs)
        mel_recalls = _extract_per_class_recall({method: data}, 0)
        nv_recalls = _extract_per_class_recall({method: data}, 1)
        mel_recall = mel_recalls[0] if mel_recalls else 0
        nv_recall = nv_recalls[0] if nv_recalls else 0
        print(f"{METHOD_LABELS[method]:<30} {avg_bacc:>10.4f} {mel_recall:>12.4f} {nv_recall:>10.4f}")

    print("-"*70)
    print("pFedST (MICCAI 2025, published):          ~0.820    (homogeneous, not reproduced)")
    print("="*70)


if __name__ == '__main__':
    results = load_results()
    if results:
        print_results_table(results)
        plot_bacc_over_rounds(results)
        plot_final_per_class_recall(results)
    else:
        print("No results found. Run experiments first.")

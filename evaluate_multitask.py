"""
Evaluate Multi-Task Model on Test Set.

Generates:
1. SHD Classification metrics (12-lead and 6-limb-lead inputs)
   - AUROC, AUPRC, accuracy, F1
   - ROC curves, PR curves, confusion matrices

2. Reconstruction quality (from 6 limb leads -> 12 leads)
   - Per-lead Pearson correlation
   - MSE per lead
   - Visual reconstruction plots for sample ECGs
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12
from ecg_gnn_v2 import ECGGNNV2
from train_multitask_shd import MultiTaskSHDModel

class EvalDataset(Dataset):
    """Dataset for evaluation with configurable lead inputs."""

    def __init__(
        self,
        waveforms_path: str,
        tabular_path: str,
        labels_path: str,
        split: str = 'test',
        lead_indices: list = None,
    ):
        self.waveforms = np.load(waveforms_path, mmap_mode='r')
        self.tabular = np.load(tabular_path).astype(np.float32)
        metadata = pd.read_csv(labels_path)
        split_mask = metadata['split'] == split
        split_data = metadata[split_mask].reset_index(drop=True)
        self.labels = split_data['shd_moderate_or_greater_flag'].values.astype(np.float32)
        self.indices = list(range(len(self.labels)))
        self.n_samples = len(self.labels)

        self.lead_indices = lead_indices if lead_indices is not None else list(range(12))
        self.builder = ECGGraphBuilderV2()

        print(f"Loaded {split} set: {self.n_samples} samples")
        print(f"  Using leads: {[LEAD_NAMES_12[i] for i in self.lead_indices]}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        ecg = self.waveforms[actual_idx]
        if ecg.ndim == 3:
            ecg = ecg[0]
        ecg = ecg.T
        ecg = np.array(ecg, dtype=np.float32)
        graph = self.builder.build_subgraph_from_indices(
            ecg, self.lead_indices, bidirectional=True
        )
        extended_ecg = self.builder.compute_extended_targets(ecg, include_negated=True)
        graph.extended_ecg = extended_ecg
        graph.full_ecg = torch.tensor(ecg, dtype=torch.float32)
        tabular = torch.tensor(self.tabular[actual_idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return graph, tabular, label

def collate_fn(batch):
    graphs, tabulars, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(tabulars),
        torch.stack(labels),
    )


@torch.no_grad()
def evaluate_classification(model, loader, query_coords, device):
    """Evaluate SHD classification performance."""
    model.eval()
    all_probs = []
    all_labels = []

    for graph_batch, tabular, labels in tqdm(loader, desc='Evaluating classification'):
        graph_batch = graph_batch.to(device)
        tabular = tabular.to(device)
        B = graph_batch.num_graphs

        outputs = model(graph_batch, tabular, query_coords)
        logits = outputs['logits'].squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    preds = (all_probs >= 0.5).astype(int)
    metrics = {
        'auroc': roc_auc_score(all_labels, all_probs),
        'auprc': average_precision_score(all_labels, all_probs),
        'accuracy': accuracy_score(all_labels, preds),
        'f1': f1_score(all_labels, preds),
        'positive_rate': all_labels.mean(),
        'predicted_positive_rate': preds.mean(),
    }

    return metrics, all_probs, all_labels


@torch.no_grad()
def evaluate_reconstruction(model, loader, query_coords, device):
    """Evaluate reconstruction quality (from partial leads to all 13)."""
    model.eval()
    all_reconstructed = []
    all_targets = []

    for graph_batch, tabular, labels in tqdm(loader, desc='Evaluating reconstruction'):
        graph_batch = graph_batch.to(device)
        tabular = tabular.to(device)
        B = graph_batch.num_graphs

        outputs = model(graph_batch, tabular, query_coords)
        reconstructed = outputs['reconstructed']

        extended_ecg = graph_batch.extended_ecg
        T = reconstructed.shape[-1]
        extended_ecg = extended_ecg.view(B, 26, T)
        target = extended_ecg[:, :13, :]

        all_reconstructed.append(reconstructed.cpu())
        all_targets.append(target.cpu())

    all_reconstructed = torch.cat(all_reconstructed, dim=0) 
    all_targets = torch.cat(all_targets, dim=0) 

    lead_names = LEAD_NAMES_12 + ['ICM']
    per_lead_metrics = {}

    for i, name in enumerate(lead_names):
        pred = all_reconstructed[:, i, :] 
        tgt = all_targets[:, i, :]
        mse = F.mse_loss(pred, tgt).item()
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        tgt_centered = tgt - tgt.mean(dim=-1, keepdim=True)
        numerator = (pred_centered * tgt_centered).sum(dim=-1)
        denominator = pred_centered.norm(dim=-1) * tgt_centered.norm(dim=-1) + 1e-8
        correlations = numerator / denominator
        mean_corr = correlations.mean().item()

        per_lead_metrics[name] = {
            'mse': mse,
            'pearson_corr': mean_corr,
        }

    return per_lead_metrics, all_reconstructed, all_targets

def plot_roc_curves(results_dict, save_path):
    """Plot ROC curves for different input configurations."""
    plt.figure(figsize=(8, 6))

    colors = {'12-lead': 'blue', '6-limb': 'orange'}

    for name, (probs, labels) in results_dict.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, color=colors.get(name, 'gray'), linewidth=2,
                 label=f'{name} (AUC = {auroc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - SHD Classification', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to {save_path}")


def plot_pr_curves(results_dict, save_path):
    """Plot Precision-Recall curves for different input configurations."""
    plt.figure(figsize=(8, 6))

    colors = {'12-lead': 'blue', '6-limb': 'orange'}

    for name, (probs, labels) in results_dict.items():
        precision, recall, _ = precision_recall_curve(labels, probs)
        auprc = average_precision_score(labels, probs)
        plt.plot(recall, precision, color=colors.get(name, 'gray'), linewidth=2,
                 label=f'{name} (AP = {auprc:.3f})')

    baseline = list(results_dict.values())[0][1].mean()
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                label=f'Baseline ({baseline:.3f})')

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - SHD Classification', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves to {save_path}")


def plot_confusion_matrices(results_dict, save_path):
    """Plot confusion matrices side by side."""
    fig, axes = plt.subplots(1, len(results_dict), figsize=(5 * len(results_dict), 4))
    if len(results_dict) == 1:
        axes = [axes]

    for ax, (name, (probs, labels)) in zip(axes, results_dict.items()):
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{name}', fontsize=12)

        classes = ['No SHD', 'SHD']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12)

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrices to {save_path}")


def plot_per_lead_reconstruction(per_lead_metrics, save_path):
    """Plot per-lead reconstruction quality."""
    lead_names = list(per_lead_metrics.keys())
    correlations = [per_lead_metrics[n]['pearson_corr'] for n in lead_names]
    mses = [per_lead_metrics[n]['mse'] for n in lead_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Correlation plot
    colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in correlations]
    bars1 = ax1.bar(lead_names, correlations, color=colors, edgecolor='black')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair (0.5)')
    ax1.set_xlabel('Lead', fontsize=12)
    ax1.set_ylabel('Pearson Correlation', fontsize=12)
    ax1.set_title('Reconstruction Quality: 6 Limb Leads -> 13 Leads', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # MSE plot
    ax2.bar(lead_names, mses, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Lead', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('Reconstruction MSE per Lead', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-lead reconstruction metrics to {save_path}")


def plot_reconstruction_examples(reconstructed, targets, dataset, num_samples=3, save_path=None):
    """Plot example reconstructions."""
    lead_names = LEAD_NAMES_12 + ['ICM']
    limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    precordial_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'ICM']

    np.random.seed(42)
    sample_indices = np.random.choice(len(reconstructed), num_samples, replace=False)

    for sample_idx in sample_indices:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(13, 1, figure=fig, hspace=0.3)

        pred = reconstructed[sample_idx].numpy()
        tgt = targets[sample_idx].numpy()

        for i, name in enumerate(lead_names):
            ax = fig.add_subplot(gs[i, 0])

            t = np.arange(pred.shape[1]) / 500
            ax.plot(t, tgt[i], 'b-', linewidth=1, alpha=0.8, label='Ground Truth')
            ax.plot(t, pred[i], 'r-', linewidth=1, alpha=0.8, label='Reconstructed')
            if name in limb_leads:
                ax.set_facecolor('#e6ffe6')

            corr = np.corrcoef(pred[i], tgt[i])[0, 1]
            ax.set_ylabel(f'{name}\nr={corr:.2f}', fontsize=9)
            ax.set_xlim(0, t[-1])
            ax.tick_params(axis='y', labelsize=8)

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
            if i < len(lead_names) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (s)', fontsize=10)

        fig.suptitle(f'ECG Reconstruction from 6 Limb Leads (Sample {sample_idx})\n'
                     f'Green background = Input leads', fontsize=14)

        if save_path:
            sample_path = save_path.replace('.pdf', f'_sample{sample_idx}.pdf')
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            print(f"Saved reconstruction example to {sample_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Multi-Task Model')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints_multitask/best_multitask_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_multitask',
                        help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_reconstruction_samples', type=int, default=5,
                        help='Number of reconstruction examples to plot')
    args = parser.parse_args()

    print("Multi-Task Model Evaluation")
    print("=" * 60)

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
    TEST_WAVEFORMS = os.path.join(DATA_DIR, 'EchoNext_test_waveforms.npy')
    TEST_TABULAR = os.path.join(DATA_DIR, 'EchoNext_test_tabular_features.npy')
    LABELS_PATH = os.path.join(DATA_DIR, 'echonext_metadata_100k.csv')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading checkpoint: {args.checkpoint}")
    gnn = ECGGNNV2(
        input_length=2500,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        latent_dim=192,
        num_gnn_layers=3,
    )
    model = MultiTaskSHDModel(gnn).to(DEVICE)

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
    print(f"  Val AUROC: {checkpoint.get('val_auroc', '?')}")

    builder = ECGGraphBuilderV2()
    query_coords_13 = builder.get_13lead_coords().to(DEVICE)

    print("\n" + "=" * 60)
    print("EVALUATION: 12-Lead Input")
    print("=" * 60)

    dataset_12lead = EvalDataset(
        TEST_WAVEFORMS, TEST_TABULAR, LABELS_PATH,
        split='test',
        lead_indices=list(range(12)),
    )
    loader_12lead = DataLoader(
        dataset_12lead, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    metrics_12, probs_12, labels_12 = evaluate_classification(
        model, loader_12lead, query_coords_13, DEVICE
    )

    print("\n12-Lead Classification Metrics:")
    print(f"  AUROC:    {metrics_12['auroc']:.4f}")
    print(f"  AUPRC:    {metrics_12['auprc']:.4f}")
    print(f"  Accuracy: {metrics_12['accuracy']:.4f}")
    print(f"  F1 Score: {metrics_12['f1']:.4f}")


    print("\n" + "=" * 60)
    print("EVALUATION: 6 Limb Leads Only (I, II, III, aVR, aVL, aVF)")
    print("=" * 60)

    limb_indices = [0, 1, 2, 3, 4, 5]  # I, II, III, aVR, aVL, aVF
    dataset_limb = EvalDataset(
        TEST_WAVEFORMS, TEST_TABULAR, LABELS_PATH,
        split='test',
        lead_indices=limb_indices,
    )
    loader_limb = DataLoader(
        dataset_limb, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    metrics_limb, probs_limb, labels_limb = evaluate_classification(
        model, loader_limb, query_coords_13, DEVICE
    )

    print("\n6-Limb-Lead Classification Metrics:")
    print(f"  AUROC:    {metrics_limb['auroc']:.4f}")
    print(f"  AUPRC:    {metrics_limb['auprc']:.4f}")
    print(f"  Accuracy: {metrics_limb['accuracy']:.4f}")
    print(f"  F1 Score: {metrics_limb['f1']:.4f}")

    print("\n" + "=" * 60)
    print("RECONSTRUCTION: 6 Limb Leads -> 13 Leads")
    print("=" * 60)

    per_lead_metrics, reconstructed, targets = evaluate_reconstruction(
        model, loader_limb, query_coords_13, DEVICE
    )

    print("\nPer-Lead Reconstruction Metrics:")
    print(f"{'Lead':<8} {'Pearson r':>10} {'MSE':>10}")
    print("-" * 30)
    for name, metrics in per_lead_metrics.items():
        print(f"{name:<8} {metrics['pearson_corr']:>10.4f} {metrics['mse']:>10.4f}")

    avg_corr = np.mean([m['pearson_corr'] for m in per_lead_metrics.values()])
    avg_mse = np.mean([m['mse'] for m in per_lead_metrics.values()])
    print("-" * 30)
    print(f"{'Average':<8} {avg_corr:>10.4f} {avg_mse:>10.4f}")

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    results_dict = {
        '12-lead': (probs_12, labels_12),
        '6-limb': (probs_limb, labels_limb),
    }

    plot_roc_curves(results_dict, os.path.join(args.output_dir, 'roc_curves.pdf'))
    plot_pr_curves(results_dict, os.path.join(args.output_dir, 'pr_curves.pdf'))
    plot_confusion_matrices(results_dict, os.path.join(args.output_dir, 'confusion_matrices.pdf'))

    plot_per_lead_reconstruction(per_lead_metrics, os.path.join(args.output_dir, 'reconstruction_quality.pdf'))
    plot_reconstruction_examples(
        reconstructed, targets, dataset_limb,
        num_samples=args.num_reconstruction_samples,
        save_path=os.path.join(args.output_dir, 'reconstruction_example.pdf')
    )

    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Multi-Task Model Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("12-Lead Input - Classification Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics_12.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\n6-Limb-Lead Input - Classification Metrics:\n")
        f.write("-" * 40 + "\n")
        for key, value in metrics_limb.items():
            f.write(f"  {key}: {value:.4f}\n")

        f.write("\n6-Limb-Lead -> 13-Lead Reconstruction:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Lead':<8} {'Pearson r':>10} {'MSE':>10}\n")
        for name, metrics in per_lead_metrics.items():
            f.write(f"{name:<8} {metrics['pearson_corr']:>10.4f} {metrics['mse']:>10.4f}\n")
        f.write(f"{'Average':<8} {avg_corr:>10.4f} {avg_mse:>10.4f}\n")

        f.write("\n\n12-Lead Classification Report:\n")
        f.write("-" * 40 + "\n")
        preds_12 = (probs_12 >= 0.5).astype(int)
        f.write(classification_report(labels_12, preds_12, target_names=['No SHD', 'SHD']))

        f.write("\n6-Limb-Lead Classification Report:\n")
        f.write("-" * 40 + "\n")
        preds_limb = (probs_limb >= 0.5).astype(int)
        f.write(classification_report(labels_limb, preds_limb, target_names=['No SHD', 'SHD']))

    print(f"\nSaved metrics to {metrics_path}")

    predictions_df = pd.DataFrame({
        'label': labels_12,
        'prob_12lead': probs_12,
        'pred_12lead': (probs_12 >= 0.5).astype(int),
        'prob_6limb': probs_limb,
        'pred_6limb': (probs_limb >= 0.5).astype(int),
    })
    predictions_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    print(f"Saved predictions to {os.path.join(args.output_dir, 'predictions.csv')}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  12-lead AUROC: {metrics_12['auroc']:.4f}")
    print(f"  6-limb  AUROC: {metrics_limb['auroc']:.4f}")
    print(f"  AUROC drop:    {metrics_12['auroc'] - metrics_limb['auroc']:.4f}")
    print(f"  Avg reconstruction correlation: {avg_corr:.4f}")


if __name__ == '__main__':
    main()

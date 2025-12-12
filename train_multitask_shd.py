"""
Multi-Task Learning for SHD Classification + ECG Reconstruction.

Architecture:
    ECG Input (with random lead dropout) -> GNN Encoder -> Latent Embedding
                                                              |
                              +-------------------------------+-------------------------------+
                              |                                                               |
                    Reconstruction Head                                           Classification Head
                    (Coordinate Decoder)                                          (MLP + Tabular Fusion)
                              |                                                               |
                    L_recon = MSE + Algebraic                                         L_cls = BCE

Loss = w_cls * L_cls + w_recon * L_recon

Extended Reconstruction Targets:
- 12 standard leads (I, II, III, aVR, aVL, aVF, V1-V6)
- 1 ICM lead (V3 - V2)
- 13 negated leads (optional, sampled with lower probability)

Benefits:
- Reconstruction regularizes the encoder (forces meaningful ECG representations)
- Lead dropout during training makes model robust to missing/partial leads
- ICM lead supervision enables ICM-specific inference
- Negated leads teach bidirectional physics
- Single training phase (no separate pretraining)
- Shared representations may generalize better
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from typing import Tuple, List, Optional

import glob
import re

from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12
from ecg_gnn_v2 import ECGGNNV2, ReconstructionLossV2


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in save_dir based on epoch number."""
    pattern = os.path.join(save_dir, 'checkpoint_epoch*.pt')
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        best_path = os.path.join(save_dir, 'best_multitask_model.pt')
        if os.path.exists(best_path):
            return best_path
        return None

    def get_epoch(path):
        match = re.search(r'checkpoint_epoch(\d+)\.pt', path)
        return int(match.group(1)) if match else 0

    latest = max(checkpoints, key=get_epoch)
    return latest


def sample_reconstruction_targets(
    include_negated_prob: float = 0.2,
    num_negated_to_sample: int = 3,
) -> Tuple[List[int], List[int]]:
    """
    Sample which leads to reconstruct during training.

    Primary targets (always included): 13 leads (12 standard + ICM)
    Secondary targets (randomly sampled): negated leads

    Args:
        include_negated_prob: Probability of including any negated leads
        num_negated_to_sample: How many negated leads to sample (if included)

    Returns:
        target_indices: Indices into the 26-lead extended target tensor
        is_negated: Boolean mask for which are negated (for potential weighting)

    Target tensor order (26 total):
        [0-11]: 12 standard leads
        [12]: ICM
        [13-24]: 12 negated standard leads
        [25]: negated ICM
    """
    target_indices = list(range(13))

    if random.random() < include_negated_prob:
        negated_indices = list(range(13, 26))
        sampled_negated = random.sample(negated_indices, min(num_negated_to_sample, len(negated_indices)))
        target_indices.extend(sampled_negated)

    return sorted(target_indices)

class MultiTaskSHDModel(nn.Module):
    """
    Multi-task model with shared GNN encoder and two heads:
    1. Reconstruction head (coordinate-based decoder from ECGGNNV2)
    2. Classification head (late fusion with tabular features)
    """

    def __init__(
        self,
        gnn: ECGGNNV2,
        tabular_dim: int = 7,
        tabular_hidden: int = 64,
        cls_hidden: int = 128,
    ):
        super().__init__()
        self.gnn = gnn
        latent_dim = gnn.latent_dim

        self.cls_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, tabular_hidden),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(tabular_hidden, tabular_hidden),
            nn.SiLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim + tabular_hidden, cls_hidden),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(cls_hidden, cls_hidden // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(cls_hidden // 2, 1),
        )

    def forward(
        self,
        graph_batch: Batch,
        tabular: torch.Tensor,
        query_coords: torch.Tensor = None,
    ):
        """
        Forward pass for both tasks.

        Args:
            graph_batch: Batched PyG graphs (potentially with dropped leads)
            tabular: (B, 7) tabular features
            query_coords: (12, 2) coordinates for reconstruction (theta, phi)

        Returns:
            dict with 'logits' for classification and 'reconstructed' for reconstruction
        """
        outputs = self.gnn(graph_batch, task='encode')
        latent_temporal = outputs['latent_temporal']

        cls_embed = self.cls_pool(latent_temporal)
        tabular_embed = self.tabular_encoder(tabular)
        fused = torch.cat([cls_embed, tabular_embed], dim=-1)
        logits = self.classifier(fused)

        reconstructed = None
        if query_coords is not None:
            reconstructed = self.gnn.decode(latent_temporal, query_coords)

        return {
            'logits': logits,
            'reconstructed': reconstructed,
            'latent_temporal': latent_temporal,
        }

class MultiTaskSHDDataset(Dataset):
    """
    Dataset for multi-task learning with lead dropout.

    For each sample:
    1. Load full 12-lead ECG + tabular features + SHD label
    2. Optionally drop leads (during training) to create partial input
    3. Return graph, tabular, label, and full ECG for reconstruction target
    """

    def __init__(
        self,
        waveforms_path: str,
        tabular_path: str,
        labels_path: str,
        split: str = 'train',
        min_leads: int = 6,
        max_leads: int = 12,
        lead_dropout_prob: float = 0.5,
    ):
        print(f"Loading waveforms from {waveforms_path}...")
        self.waveforms = np.load(waveforms_path, mmap_mode='r')
        print(f"  Shape: {self.waveforms.shape}")

        print(f"Loading tabular from {tabular_path}...")
        self.tabular = np.load(tabular_path).astype(np.float32)
        print(f"  Shape: {self.tabular.shape}")

        print(f"Loading labels from {labels_path}...")
        metadata = pd.read_csv(labels_path)
        split_mask = metadata['split'] == split
        split_data = metadata[split_mask].reset_index(drop=True)

        self.labels = split_data['shd_moderate_or_greater_flag'].values.astype(np.float32)
        self.indices = list(range(len(self.labels)))

        self.n_samples = len(self.labels)
        pos_rate = self.labels.mean()
        print(f"  {split} set: {self.n_samples} samples, positive rate: {pos_rate:.2%}")

        self.min_leads = min_leads
        self.max_leads = max_leads
        self.lead_dropout_prob = lead_dropout_prob
        self.is_train = (split == 'train')

        self.builder = ECGGraphBuilderV2()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]

        ecg = self.waveforms[actual_idx]
        if ecg.ndim == 3:
            ecg = ecg[0]
        ecg = ecg.T 
        ecg = np.array(ecg, dtype=np.float32)

        if self.is_train and random.random() < self.lead_dropout_prob:
            num_leads = random.randint(self.min_leads, self.max_leads)
            observed_idx = sorted(random.sample(range(12), num_leads))
        else:
            observed_idx = list(range(12))

        graph = self.builder.build_subgraph_from_indices(
            ecg, observed_idx, bidirectional=True
        )

        extended_ecg = self.builder.compute_extended_targets(ecg, include_negated=True)
        graph.extended_ecg = extended_ecg

        tabular = torch.tensor(self.tabular[actual_idx], dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return graph, tabular, label


def collate_fn(batch):
    """Collate graphs, tabular, and labels."""
    graphs, tabulars, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(tabulars),
        torch.stack(labels),
    )

def train_epoch(
    model: MultiTaskSHDModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    recon_loss_fn: ReconstructionLossV2,
    all_query_coords: torch.Tensor,
    device: str,
    cls_weight: float = 1.0,
    recon_weight: float = 0.1,
    include_negated_prob: float = 0.2,
    num_negated_to_sample: int = 3,
):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_recon_loss = 0
    total_pearson = 0
    all_probs = []
    all_labels = []
    num_batches = 0

    pbar = tqdm(loader, desc='Training')
    for graph_batch, tabular, labels in pbar:
        graph_batch = graph_batch.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        B = graph_batch.num_graphs

        optimizer.zero_grad()

        target_indices = sample_reconstruction_targets(
            include_negated_prob=include_negated_prob,
            num_negated_to_sample=num_negated_to_sample,
        )
        query_coords = all_query_coords[target_indices]

        outputs = model(graph_batch, tabular, query_coords)
        logits = outputs['logits'].squeeze(-1)
        reconstructed = outputs['reconstructed']

        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        extended_ecg = graph_batch.extended_ecg 
        T = reconstructed.shape[-1]
        N_sampled = len(target_indices)

        extended_ecg = extended_ecg.view(B, 26, T)
        target_ecg = extended_ecg[:, target_indices, :] 

        recon_loss = F.mse_loss(reconstructed, target_ecg)

        with torch.no_grad():
            pred_flat = reconstructed.reshape(-1, T)
            tgt_flat = target_ecg.reshape(-1, T)
            pred_centered = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
            tgt_centered = tgt_flat - tgt_flat.mean(dim=-1, keepdim=True)
            numerator = (pred_centered * tgt_centered).sum(dim=-1)
            denominator = pred_centered.norm(dim=-1) * tgt_centered.norm(dim=-1) + 1e-8
            pearson_corr = (numerator / denominator).mean()

        loss = cls_weight * cls_loss + recon_weight * recon_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_recon_loss += recon_loss.item()
        total_pearson += pearson_corr.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{total_loss/num_batches:.4f}',
            'cls': f'{total_cls_loss/num_batches:.4f}',
            'recon': f'{total_recon_loss/num_batches:.4f}',
            'corr': f'{total_pearson/num_batches:.4f}',
        })

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'pearson_corr': total_pearson / num_batches,
        'auroc': auroc,
        'auprc': auprc,
    }


@torch.no_grad()
def validate(
    model: MultiTaskSHDModel,
    loader: DataLoader,
    recon_loss_fn: ReconstructionLossV2,
    all_query_coords: torch.Tensor,
    device: str,
    cls_weight: float = 1.0,
    recon_weight: float = 0.1,
):
    """Validate on primary 13 leads (12 standard + ICM) for consistent evaluation."""
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_recon_loss = 0
    total_pearson = 0
    all_probs = []
    all_labels = []
    num_batches = 0

    primary_indices = list(range(13))
    query_coords = all_query_coords[primary_indices]

    for graph_batch, tabular, labels in tqdm(loader, desc='Validating'):
        graph_batch = graph_batch.to(device)
        tabular = tabular.to(device)
        labels = labels.to(device)
        B = graph_batch.num_graphs

        outputs = model(graph_batch, tabular, query_coords)
        logits = outputs['logits'].squeeze(-1)
        reconstructed = outputs['reconstructed']

        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)

        extended_ecg = graph_batch.extended_ecg
        T = reconstructed.shape[-1]
        extended_ecg = extended_ecg.view(B, 26, T)
        target_ecg = extended_ecg[:, primary_indices, :] 

        recon_loss = F.mse_loss(reconstructed, target_ecg)

        pred_flat = reconstructed.reshape(-1, T)
        tgt_flat = target_ecg.reshape(-1, T)
        pred_centered = pred_flat - pred_flat.mean(dim=-1, keepdim=True)
        tgt_centered = tgt_flat - tgt_flat.mean(dim=-1, keepdim=True)
        numerator = (pred_centered * tgt_centered).sum(dim=-1)
        denominator = pred_centered.norm(dim=-1) * tgt_centered.norm(dim=-1) + 1e-8
        pearson_corr = (numerator / denominator).mean()

        loss = cls_weight * cls_loss + recon_weight * recon_loss

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_recon_loss += recon_loss.item()
        total_pearson += pearson_corr.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    return {
        'loss': total_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
        'recon_loss': total_recon_loss / num_batches,
        'pearson_corr': total_pearson / num_batches,
        'auroc': auroc,
        'auprc': auprc,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-Task SHD Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cls_weight', type=float, default=1.0,
                        help='Weight for classification loss')
    parser.add_argument('--recon_weight', type=float, default=0.1,
                        help='Weight for reconstruction loss')
    parser.add_argument('--min_leads', type=int, default=6,
                        help='Minimum leads to keep during dropout')
    parser.add_argument('--max_leads', type=int, default=12,
                        help='Maximum leads (12 = no dropout)')
    parser.add_argument('--lead_dropout_prob', type=float, default=0.5,
                        help='Probability of applying lead dropout during training')
    parser.add_argument('--include_negated_prob', type=float, default=0.2,
                        help='Probability of including negated leads in reconstruction')
    parser.add_argument('--num_negated_to_sample', type=int, default=3,
                        help='Number of negated leads to sample when included')
    parser.add_argument('--save_dir', type=str, default='checkpoints_multitask')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained GNN checkpoint (optional)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint in save_dir')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from specific checkpoint path')
    args = parser.parse_args()

    print("Multi-Task SHD Classification + Reconstruction")
    print("=" * 60)
    print(f"Classification weight: {args.cls_weight}")
    print(f"Reconstruction weight: {args.recon_weight}")
    print(f"Lead dropout: {args.min_leads}-{args.max_leads} leads, prob={args.lead_dropout_prob}")
    print(f"Reconstruction targets: 13 primary + negated (prob={args.include_negated_prob}, n={args.num_negated_to_sample})")

    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset')
    TRAIN_WAVEFORMS = os.path.join(DATA_DIR, 'EchoNext_train_waveforms.npy')
    VAL_WAVEFORMS = os.path.join(DATA_DIR, 'EchoNext_val_waveforms.npy')
    TRAIN_TABULAR = os.path.join(DATA_DIR, 'EchoNext_train_tabular_features.npy')
    VAL_TABULAR = os.path.join(DATA_DIR, 'EchoNext_val_tabular_features.npy')
    LABELS_PATH = os.path.join(DATA_DIR, 'echonext_metadata_100k.csv')

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}")

    print("\nLoading datasets...")
    train_dataset = MultiTaskSHDDataset(
        TRAIN_WAVEFORMS, TRAIN_TABULAR, LABELS_PATH,
        split='train',
        min_leads=args.min_leads,
        max_leads=args.max_leads,
        lead_dropout_prob=args.lead_dropout_prob,
    )
    val_dataset = MultiTaskSHDDataset(
        VAL_WAVEFORMS, VAL_TABULAR, LABELS_PATH,
        split='val',
        min_leads=12, max_leads=12,
        lead_dropout_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nCreating model...")
    gnn = ECGGNNV2(
        input_length=2500,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        latent_dim=192,
        num_gnn_layers=3,
    )

    if args.pretrained:
        print(f"Loading pretrained GNN from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu', weights_only=False)
        gnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("  Loaded pretrained weights")

    model = MultiTaskSHDModel(gnn).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    start_epoch = 1
    best_val_auroc = 0.0

    if args.resume or args.resume_from:
        if args.resume_from:
            resume_path = args.resume_from
        else:
            resume_path = find_latest_checkpoint(args.save_dir)

        if resume_path and os.path.exists(resume_path):
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auroc = checkpoint.get('val_auroc', 0.0)

            for _ in range(checkpoint['epoch']):
                scheduler.step()

            print(f"  Resumed from epoch {checkpoint['epoch']}")
            print(f"  Best val AUROC so far: {best_val_auroc:.4f}")
            print(f"  Continuing from epoch {start_epoch}")
        else:
            print(f"\nNo checkpoint found to resume from in {args.save_dir}")

    recon_loss_fn = ReconstructionLossV2()

    builder = ECGGraphBuilderV2()
    all_query_coords, lead_names = builder.get_extended_coords(include_negated=True)
    all_query_coords = all_query_coords.to(DEVICE)
    print(f"Extended reconstruction targets: {len(lead_names)} leads")
    print(f"  Primary (0-12): {lead_names[:13]}")
    print(f"  Negated (13-25): {lead_names[13:]}")

    os.makedirs(args.save_dir, exist_ok=True)

    print("\nStarting training...")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, recon_loss_fn, all_query_coords, DEVICE,
            cls_weight=args.cls_weight, recon_weight=args.recon_weight,
            include_negated_prob=args.include_negated_prob,
            num_negated_to_sample=args.num_negated_to_sample,
        )
        val_metrics = validate(
            model, val_loader, recon_loss_fn, all_query_coords, DEVICE,
            cls_weight=args.cls_weight, recon_weight=args.recon_weight,
        )

        scheduler.step()

        print(f"  Train - loss: {train_metrics['loss']:.4f}, "
              f"cls: {train_metrics['cls_loss']:.4f}, recon: {train_metrics['recon_loss']:.4f}, "
              f"corr: {train_metrics['pearson_corr']:.4f}")
        print(f"        AUROC: {train_metrics['auroc']:.4f}, AUPRC: {train_metrics['auprc']:.4f}")
        print(f"  Val   - loss: {val_metrics['loss']:.4f}, "
              f"cls: {val_metrics['cls_loss']:.4f}, recon: {val_metrics['recon_loss']:.4f}, "
              f"corr: {val_metrics['pearson_corr']:.4f}")
        print(f"        AUROC: {val_metrics['auroc']:.4f}, AUPRC: {val_metrics['auprc']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gnn_state_dict': model.gnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_metrics['auroc'],
                'val_auprc': val_metrics['auprc'],
                'val_loss': val_metrics['loss'],
            }, os.path.join(args.save_dir, 'best_multitask_model.pt'))
            print(f"  Saved new best model (val_auroc={best_val_auroc:.4f})")

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'gnn_state_dict': model.gnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auroc': val_metrics['auroc'],
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))

    print("\nTraining complete!")
    print(f"Best val AUROC: {best_val_auroc:.4f}")


if __name__ == '__main__':
    main()

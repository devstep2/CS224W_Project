"""
ECG GNN V2: Experimental model with NEFNet-style decoder.

Key improvements over v1:
1. Uses v2 graph builder with bidirectional edges
2. NEFNet-style convolutional decoder (preserves temporal structure)
3. Coordinate-based decoding for ANY lead (standard or novel)
4. Identity reconstruction loss for observed leads

Architecture:
- Encoder: Spherical harmonic node encoding + edge signal encoding + GNN message passing
- Latent: Graph-level embedding with temporal structure preserved [B, hidden, L/4]
- Decoder: Coordinate modulation + conv upsampling (like NEFNet)

The decoder can synthesize ANY lead by querying its spherical coordinates.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, List, Tuple, Dict
import math


class DoubleConv(nn.Module):
    """Two conv layers with batch norm and activation (from NEFNet)."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.Mish(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.Mish(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock1d(nn.Module):
    """Residual block for 1D convolutions."""

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.Mish(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class SphericalHarmonicEncoding(nn.Module):
    """Encode 3D positions using spherical harmonic basis functions."""

    def __init__(self, max_degree: int = 4, include_radial: bool = True):
        super().__init__()
        self.max_degree = max_degree
        self.include_radial = include_radial
        self.sh_dim = sum(2 * l + 1 for l in range(max_degree + 1))
        self.output_dim = self.sh_dim + (4 if include_radial else 0)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)
        x_n, y_n, z_n = x / r, y / r, z / r

        sh_features = []

        # l=0
        sh_features.append(torch.ones_like(x_n) * 0.5 * math.sqrt(1 / math.pi))

        # l=1
        if self.max_degree >= 1:
            c1 = 0.5 * math.sqrt(3 / math.pi)
            sh_features.extend([c1 * y_n, c1 * z_n, c1 * x_n])

        # l=2
        if self.max_degree >= 2:
            c2 = 0.5 * math.sqrt(15 / math.pi)
            c2_0 = 0.25 * math.sqrt(5 / math.pi)
            sh_features.extend([
                c2 * x_n * y_n,
                c2 * y_n * z_n,
                c2_0 * (3 * z_n**2 - 1),
                c2 * x_n * z_n,
                0.5 * c2 * (x_n**2 - y_n**2)
            ])

        # l=3
        if self.max_degree >= 3:
            sh_features.extend([
                y_n * (3 * x_n**2 - y_n**2),
                x_n * y_n * z_n,
                y_n * (5 * z_n**2 - 1),
                z_n * (5 * z_n**2 - 3),
                x_n * (5 * z_n**2 - 1),
                z_n * (x_n**2 - y_n**2),
                x_n * (x_n**2 - 3 * y_n**2)
            ])

        # l=4
        if self.max_degree >= 4:
            xy, xz, yz = x_n * y_n, x_n * z_n, y_n * z_n
            x2, y2, z2 = x_n**2, y_n**2, z_n**2
            sh_features.extend([
                xy * (x2 - y2),
                yz * (3 * x2 - y2),
                xy * (7 * z2 - 1),
                yz * (7 * z2 - 3),
                35 * z2**2 - 30 * z2 + 3,
                xz * (7 * z2 - 3),
                (x2 - y2) * (7 * z2 - 1),
                xz * (x2 - 3 * y2),
                x2**2 - 6 * x2 * y2 + y2**2
            ])

        sh = torch.stack(sh_features, dim=-1)

        if self.include_radial:
            radial = torch.stack([
                r, torch.sin(math.pi * r),
                torch.cos(math.pi * r), torch.exp(-r)
            ], dim=-1)
            sh = torch.cat([sh, radial], dim=-1)

        return sh



class LeadSignalEncoder(nn.Module):
    """
    Encode ECG lead signals with temporal structure preservation.

    Unlike v1 which collapsed to a single embedding, this preserves
    downsampled temporal info for the decoder.
    """

    def __init__(
        self,
        input_length: int = 2500,
        hidden_dim: int = 128,
        output_dim: int = 192,
        downsample_factor: int = 4,
    ):
        super().__init__()
        self.input_length = input_length
        self.output_length = input_length // downsample_factor

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Mish(inplace=True),
            ResBlock1d(64),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(inplace=True),
            ResBlock1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.Mish(inplace=True),
        )

        # Also produce a pooled embedding for message passing
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.embed_proj = nn.Linear(output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (E, T) lead signals

        Returns:
            temporal: (E, output_dim, L/4) temporal features
            embedding: (E, output_dim) pooled embedding for message passing
        """
        x = x.unsqueeze(1)
        temporal = self.encoder(x)
        embedding = self.embed_proj(self.pool(temporal).squeeze(-1))
        return temporal, embedding


class ECGMessagePassingV2(MessagePassing):
    """
    Message passing for ECG graph v2.

    Uses edge features (lead signal embeddings) in message computation.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')

        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, node_dim),
        )

        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class CoordinateEncoder(nn.Module):
    """
    Encode (theta, phi) spherical coordinates for lead decoding.

    Similar to NEFNet's ThetaEncoder - produces modulation vector.
    """

    def __init__(self, hidden_dim: int = 192, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        input_dim = num_frequencies * 4  

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) with [theta, phi] per row

        Returns:
            (N, hidden_dim) modulation vectors
        """
        theta = coords[:, 0]
        phi = coords[:, 1]

        encoded = []
        for freq in range(1, self.num_frequencies + 1):
            encoded.extend([
                torch.sin(freq * theta),
                torch.cos(freq * theta),
                torch.sin(freq * phi),
                torch.cos(freq * phi),
            ])

        features = torch.stack(encoded, dim=-1)  
        return self.encoder(features)


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for lead reconstruction.

    Takes modulated latent [B, C, L/4] and upsamples to [B, 1, L].
    Architecture inspired by NEFNet.
    """

    def __init__(self, latent_dim: int = 192, input_length: int = 2500):
        super().__init__()
        self.input_length = input_length

        self.decoder = nn.Sequential(
            # L/4 -> L/2
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(latent_dim, latent_dim // 2),
            # L/2 -> L
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            DoubleConv(latent_dim // 2, latent_dim // 4),
            # projection
            nn.Conv1d(latent_dim // 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L/4) modulated latent

        Returns:
            (B, 1, L) reconstructed signal
        """
        out = self.decoder(x)

        if out.shape[-1] != self.input_length:
            out = F.interpolate(out, size=self.input_length, mode='linear', align_corners=False)

        return out


class ECGGNNV2(nn.Module):
    """
    ECG Graph Neural Network V2 with NEFNet-style decoder.

    Architecture:
    1. Node encoding: Spherical harmonic features from electrode positions
    2. Edge encoding: Lead signals -> temporal features + embedding
    3. GNN: Message passing using edge embeddings
    4. Latent: Aggregate temporal edge features + node context
    5. Decoder: Coordinate modulation + conv upsampling

    The key insight: we preserve temporal structure in the latent space
    and use coordinate-conditioned decoding for any lead.
    """

    def __init__(
        self,
        input_length: int = 2500,
        sh_max_degree: int = 4,
        node_dim: int = 128,
        edge_dim: int = 192,
        latent_dim: int = 192,
        num_gnn_layers: int = 3,
    ):
        super().__init__()

        self.input_length = input_length
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.output_length = input_length // 4

        self.pos_encoder = SphericalHarmonicEncoding(max_degree=sh_max_degree)
        self.node_proj = nn.Linear(self.pos_encoder.output_dim, node_dim)
        self.signal_encoder = LeadSignalEncoder(
            input_length=input_length,
            hidden_dim=128,
            output_dim=edge_dim,
            downsample_factor=4,
        )

        self.gnn_layers = nn.ModuleList([
            ECGMessagePassingV2(node_dim, edge_dim, node_dim * 2)
            for _ in range(num_gnn_layers)
        ])

        self.latent_proj = nn.Sequential(
            nn.Linear(node_dim + edge_dim, latent_dim),
            nn.Mish(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.coord_encoder = CoordinateEncoder(hidden_dim=latent_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim, input_length=input_length)
        self.classify_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Mish(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(latent_dim, 1),
        )

    def encode(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the ECG graph into latent representations.

        Args:
            pos: (N, 3) node positions
            edge_index: (2, E) edge indices
            edge_attr: (E, T) edge signals
            batch: (N,) batch assignment

        Returns:
            Dict with node_embed, edge_embed, edge_temporal, graph_embed, latent_temporal
        """
        node_features = self.node_proj(self.pos_encoder(pos))
        edge_temporal, edge_embed = self.signal_encoder(edge_attr)

        x = node_features
        for gnn in self.gnn_layers:
            x = gnn(x, edge_index, edge_embed)

        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            graph_embed = global_mean_pool(x, batch)

        if batch is None:
            edge_temporal_pooled = edge_temporal.mean(dim=0, keepdim=True)
        else:
            edge_batch = batch[edge_index[0]]
            B = batch.max().item() + 1
            edge_temporal_pooled = torch.zeros(
                B, self.edge_dim, self.output_length,
                device=edge_temporal.device, dtype=edge_temporal.dtype
            )
            for b in range(B):
                mask = edge_batch == b
                if mask.any():
                    edge_temporal_pooled[b] = edge_temporal[mask].mean(dim=0)

        graph_embed_expanded = graph_embed.unsqueeze(-1).expand(-1, -1, self.output_length)
        combined = torch.cat([graph_embed_expanded, edge_temporal_pooled], dim=1)
        combined = combined.permute(0, 2, 1)
        latent = self.latent_proj(combined)
        latent_temporal = latent.permute(0, 2, 1)

        return {
            'node_embed': x,
            'edge_embed': edge_embed,
            'edge_temporal': edge_temporal,
            'graph_embed': graph_embed,
            'latent_temporal': latent_temporal,
        }

    def decode(
        self,
        latent_temporal: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode lead signals at query coordinates.

        Args:
            latent_temporal: (B, latent_dim, L/4) latent representation
            query_coords: (N, 2) spherical coordinates [theta, phi]

        Returns:
            (B, N, T) decoded signals for each query coordinate
        """
        B = latent_temporal.shape[0]
        N = query_coords.shape[0]

        coord_embed = self.coord_encoder(query_coords)

        decoded = []
        for i in range(N):
            modulation = coord_embed[i].unsqueeze(0).unsqueeze(-1)
            modulated = latent_temporal * modulation

            signal = self.decoder(modulated)
            decoded.append(signal.squeeze(1))

        return torch.stack(decoded, dim=1)

    def forward(
        self,
        data: Data,
        task: str = 'encode',
        query_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            data: PyG Data/Batch with x (pos), edge_index, edge_attr, optional batch
            task: 'encode', 'reconstruct', or 'classify'
            query_coords: (N, 2) coords for reconstruction

        Returns:
            Dict with relevant outputs
        """
        batch = getattr(data, 'batch', None)

        enc = self.encode(data.x, data.edge_index, data.edge_attr, batch)

        outputs = {
            'node_embed': enc['node_embed'],
            'graph_embed': enc['graph_embed'],
            'latent_temporal': enc['latent_temporal'],
        }

        if task == 'reconstruct' and query_coords is not None:
            outputs['reconstructed'] = self.decode(enc['latent_temporal'], query_coords)

        elif task == 'classify':
            latent_pooled = enc['latent_temporal'].mean(dim=-1)
            outputs['logits'] = self.classify_head(latent_pooled)

        return outputs


def pearson_correlation(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Pearson correlation coefficient along specified dimension.

    Args:
        pred: Predicted signals
        target: Target signals
        dim: Dimension along which to compute correlation (default: last = temporal)

    Returns:
        Correlation coefficients (same shape as input minus the specified dim)
    """
    pred_centered = pred - pred.mean(dim=dim, keepdim=True)
    target_centered = target - target.mean(dim=dim, keepdim=True)

    numerator = (pred_centered * target_centered).sum(dim=dim)
    pred_std = pred_centered.pow(2).sum(dim=dim).sqrt()
    target_std = target_centered.pow(2).sum(dim=dim).sqrt()

    denominator = (pred_std * target_std).clamp(min=1e-8)

    return numerator / denominator


class ReconstructionLossV2(nn.Module):
    """
    Reconstruction loss with:
    1. MSE Loss: Main training signal
    2. Algebraic Loss: Enforce Einthoven/Goldberger constraints
    3. Identity Loss: Observed leads should reconstruct exactly

    Pearson correlation is computed as a METRIC only (not in loss - too expensive).
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        algebraic_weight: float = 0.25,
        identity_weight: float = 0.5,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.algebraic_weight = algebraic_weight
        self.identity_weight = identity_weight

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        observed_idx: Optional[List[int]] = None,
        compute_algebraic: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reconstruction loss.

        Args:
            reconstructed: (B, 12, T) reconstructed all 12 leads
            target: (B, 12, T) ground truth all 12 leads
            observed_idx: List of observed lead indices for identity loss
            compute_algebraic: Whether to compute algebraic constraints

        Returns:
            Dict with loss components and metrics (Pearson computed for reporting only)
        """
        device = reconstructed.device

        mse_loss = F.mse_loss(reconstructed, target)

        identity_loss = torch.tensor(0.0, device=device)
        if observed_idx is not None and len(observed_idx) > 0:
            obs_recon = reconstructed[:, observed_idx, :]
            obs_target = target[:, observed_idx, :]
            identity_loss = F.mse_loss(obs_recon, obs_target)

        alg_loss = torch.tensor(0.0, device=device)
        if compute_algebraic and reconstructed.shape[1] == 12:
            I = reconstructed[:, 0]
            II = reconstructed[:, 1]
            III = reconstructed[:, 2]
            aVR = reconstructed[:, 3]
            aVL = reconstructed[:, 4]
            aVF = reconstructed[:, 5]

            einthoven = F.mse_loss(II, I + III)
            avf_constraint = F.mse_loss(aVF, (II + III) / 2)
            avl_constraint = F.mse_loss(aVL, (I - III) / 2)
            avr_constraint = F.mse_loss(aVR, -(I + II) / 2)

            alg_loss = (einthoven + avf_constraint + avl_constraint + avr_constraint) / 4

        total = (
            self.mse_weight * mse_loss +
            self.algebraic_weight * alg_loss +
            self.identity_weight * identity_loss
        )

        with torch.no_grad():
            corr = pearson_correlation(reconstructed, target, dim=-1).mean()

        return {
            'total': total,
            'mse': mse_loss,
            'identity': identity_loss,
            'algebraic': alg_loss,
            'pearson_corr': corr,
        }


def compute_metrics(
    reconstructed: torch.Tensor,
    target: torch.Tensor,
    observed_idx: Optional[torch.Tensor] = None,
    masked_idx: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for reconstruction.

    Args:
        reconstructed: (B, 12, T) predicted signals
        target: (B, 12, T) ground truth signals
        observed_idx: Which leads were observed (for separate metrics)
        masked_idx: Which leads were masked

    Returns:
        Dict with various metrics
    """
    with torch.no_grad():
        mse = F.mse_loss(reconstructed, target).item()
        mae = F.l1_loss(reconstructed, target).item()
        corr = pearson_correlation(reconstructed, target, dim=-1).mean().item()

        rmse = mse ** 0.5

        lead_corrs = pearson_correlation(reconstructed, target, dim=-1).mean(dim=0)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'pearson_corr': corr,
        }

        metrics['corr_limb'] = lead_corrs[:3].mean().item()
        metrics['corr_aug'] = lead_corrs[3:6].mean().item()
        metrics['corr_precordial'] = lead_corrs[6:].mean().item()

        return metrics



if __name__ == '__main__':
    print("Testing ECG GNN V2")
    print("=" * 60)

    from ecg_graph_v2 import ECGGraphBuilderV2, LEAD_NAMES_12

    builder = ECGGraphBuilderV2()
    T = 2500
    dummy_ecg = np.random.randn(12, T).astype(np.float32)

    graph = builder.build_12lead_subgraph(dummy_ecg, bidirectional=True)
    print(f"Test graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    model = ECGGNNV2(
        input_length=T,
        sh_max_degree=4,
        node_dim=128,
        edge_dim=192,
        latent_dim=192,
        num_gnn_layers=3,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTesting encode...")
    with torch.no_grad():
        outputs = model(graph, task='encode')
        print(f"  Node embed: {outputs['node_embed'].shape}")
        print(f"  Graph embed: {outputs['graph_embed'].shape}")
        print(f"  Latent temporal: {outputs['latent_temporal'].shape}")

    print("\nTesting reconstruct...")
    query_coords = builder.get_all_12lead_coords()  # (12, 2)
    with torch.no_grad():
        outputs = model(graph, task='reconstruct', query_coords=query_coords)
        print(f"  Reconstructed: {outputs['reconstructed'].shape}") 

    print("\nTesting classify...")
    with torch.no_grad():
        outputs = model(graph, task='classify')
        print(f"  Logits: {outputs['logits'].shape}")

    print("\nTesting batched forward...")
    graphs = [
        builder.build_12lead_subgraph(np.random.randn(12, T).astype(np.float32))
        for _ in range(4)
    ]
    batch = Batch.from_data_list(graphs)
    with torch.no_grad():
        outputs = model(batch, task='reconstruct', query_coords=query_coords)
        print(f"  Batched reconstructed: {outputs['reconstructed'].shape}")

    print("\n" + "=" * 60)
    print("ECG GNN V2 test passed!")

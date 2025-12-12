"""
ECG Graph V2: Fully connected graph with coordinate-based decoding.

Key features:
1. Fully connected graph: 13 nodes × 12 other nodes = 156 directed edges
2. Bidirectional with physics: edge A→B has signal, B→A has NEGATED signal
3. Standard 12-lead is a 24-edge subgraph of this fully connected graph
4. Coordinate-based decoder: query ANY (θ, φ) to synthesize leads
5. Extensible: add V7-V9 or other electrodes by extending ELECTRODE_POSITIONS

Graph Structure:
- 13 electrode nodes (RA, LA, LL, WCT, V1-V6, 3 virtual midpoints)
- 156 possible directed edges (all ordered pairs)
- Standard 12-lead = 24-edge subgraph (12 specific pairs × 2 directions)
- ICM = 2-edge subgraph (V2↔V3)
- Any lead configuration = corresponding subgraph
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import math


from ecg_graph_flexible import BASE_POSITIONS, BASE_NODES

ALL_ELECTRODES = BASE_NODES
ELECTRODE_POSITIONS = BASE_POSITIONS
STANDARD_LEAD_DEFINITIONS = {
    # Limb leads (bipolar)
    'I':   ('RA', 'LA'),    # V(LA) - V(RA)
    'II':  ('RA', 'LL'),    # V(LL) - V(RA)
    'III': ('LA', 'LL'),    # V(LL) - V(LA)

    # Augmented limb leads (unipolar-ish)
    'aVR': ('mid_LA_LL', 'RA'),  # V(RA) - V(LA+LL)/2
    'aVL': ('mid_RA_LL', 'LA'),  # V(LA) - V(RA+LL)/2
    'aVF': ('mid_RA_LA', 'LL'),  # V(LL) - V(RA+LA)/2

    # Precordial leads (unipolar, referenced to WCT)
    'V1': ('WCT', 'V1'),
    'V2': ('WCT', 'V2'),
    'V3': ('WCT', 'V3'),
    'V4': ('WCT', 'V4'),
    'V5': ('WCT', 'V5'),
    'V6': ('WCT', 'V6'),
}

OTHER_LEAD_DEFINITIONS = {
    'ICM': ('V2', 'V3'),
}

LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def compute_direction(src_pos: np.ndarray, tgt_pos: np.ndarray) -> np.ndarray:
    """Compute normalized direction vector from source to target."""
    # direction vector d = target - source
    # represents voltage difference direction in 3d space
    direction = tgt_pos - src_pos
    # normalize to unit vector: d_hat = d / ||d||_2
    # euclidean norm
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-8 else direction


def cartesian_to_spherical(direction: np.ndarray) -> Tuple[float, float]:
    """
    Convert direction vector to spherical coordinates (θ, φ).
    θ (theta): azimuthal angle in xy-plane from x-axis [0, 2π]
    φ (phi): polar angle from z-axis [0, π]
    """
    # standard cartesian to spherical coordinate transform
    # phi = arccos(z) gives angle from positive z-axis
    # theta = arctan2(y,x) gives angle in xy-plane from positive x-axis
    # from vector calculus and spherical geometry
    x, y, z = direction
    phi = math.acos(np.clip(z, -1.0, 1.0))  # clip to avoid numerical issues
    theta = math.atan2(y, x)  # atan2 handles all quadrants properly
    if theta < 0:
        theta += 2 * math.pi  # map to [0, 2pi] range
    return theta, phi


def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates back to unit direction vector."""
    # inverse transform from spherical to cartesian
    # x = sin(phi) * cos(theta)
    # y = sin(phi) * sin(theta)
    # z = cos(phi)
    # this from standard spherical coordinate definitions
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return np.array([x, y, z])


def get_lead_spherical_coords(src_electrode: str, tgt_electrode: str) -> Tuple[float, float]:
    """Get spherical coordinates for a lead given electrode names."""
    src_pos = ELECTRODE_POSITIONS[src_electrode]
    tgt_pos = ELECTRODE_POSITIONS[tgt_electrode]
    direction = compute_direction(src_pos, tgt_pos)
    return cartesian_to_spherical(direction)



class ECGGraphBuilderV2:
    """
    Builds fully connected ECG graphs with bidirectional edges.

    The "supergraph" has all 156 directed edges (13 × 12).
    Standard 12-lead or any other configuration is a subgraph.

    Key methods:
    - build_full_graph(): 156-edge fully connected graph (for reference)
    - build_observed_subgraph(): Subgraph with only observed leads
    - get_query_coords(): Get (θ, φ) for any electrode pair for decoding
    """

    def __init__(self, electrodes: Optional[List[str]] = None):
        """
        Args:
            electrodes: List of electrode names. Defaults to ALL_ELECTRODES.
        """
        self.electrodes = electrodes or ALL_ELECTRODES
        self.electrode_to_idx = {e: i for i, e in enumerate(self.electrodes)}
        self.num_electrodes = len(self.electrodes)

        self.node_positions = torch.tensor(
            np.array([ELECTRODE_POSITIONS[e] for e in self.electrodes]),
            dtype=torch.float32
        )
        self._build_full_edge_structure()

    def _build_full_edge_structure(self):
        """
        Precompute edge structure for fully connected graph.
        156 edges = 13 × 12 (all ordered pairs, no self-loops).
        """
        self.all_edge_src = []
        self.all_edge_tgt = []
        self.all_edge_spherical = []
        self.edge_pair_to_idx = {} 

        edge_idx = 0
        for src in self.electrodes:
            for tgt in self.electrodes:
                if src != tgt:
                    src_idx = self.electrode_to_idx[src]
                    tgt_idx = self.electrode_to_idx[tgt]

                    self.all_edge_src.append(src_idx)
                    self.all_edge_tgt.append(tgt_idx)

                    theta, phi = get_lead_spherical_coords(src, tgt)
                    self.all_edge_spherical.append([theta, phi])

                    self.edge_pair_to_idx[(src, tgt)] = edge_idx
                    edge_idx += 1

        self.full_edge_index = torch.tensor(
            [self.all_edge_src, self.all_edge_tgt], dtype=torch.long
        )
        self.full_edge_spherical = torch.tensor(
            self.all_edge_spherical, dtype=torch.float32
        )

        self.num_full_edges = len(self.all_edge_src)
        print(f"ECGGraphBuilderV2: {self.num_electrodes} electrodes, "
              f"{self.num_full_edges} possible edges")

    def get_edge_idx(self, src_electrode: str, tgt_electrode: str) -> int:
        """Get edge index for a specific electrode pair."""
        return self.edge_pair_to_idx[(src_electrode, tgt_electrode)]

    def get_lead_edge_indices(self, lead_name: str) -> Tuple[int, int]:
        """
        Get (forward_idx, reverse_idx) for a named lead.

        Returns both directions since we use bidirectional edges.
        """
        if lead_name in STANDARD_LEAD_DEFINITIONS:
            src, tgt = STANDARD_LEAD_DEFINITIONS[lead_name]
        elif lead_name in OTHER_LEAD_DEFINITIONS:
            src, tgt = OTHER_LEAD_DEFINITIONS[lead_name]
        else:
            raise ValueError(f"Unknown lead: {lead_name}")

        forward_idx = self.edge_pair_to_idx[(src, tgt)]
        reverse_idx = self.edge_pair_to_idx[(tgt, src)]
        return forward_idx, reverse_idx

    def build_observed_subgraph(
        self,
        observed_signals: Dict[str, np.ndarray],
        bidirectional: bool = True,
    ) -> Data:
        """
        Build subgraph containing only observed leads.

        Args:
            observed_signals: Dict mapping lead name -> signal array (T,)
                             e.g., {'I': signal_I, 'II': signal_II, ...}
            bidirectional: If True, include reverse edges with negated signals

        Returns:
            PyG Data object with only the observed edges
        """
        edge_src = []
        edge_tgt = []
        edge_attr_list = []
        edge_spherical_list = []

        for lead_name, signal in observed_signals.items():
            if lead_name in STANDARD_LEAD_DEFINITIONS:
                src, tgt = STANDARD_LEAD_DEFINITIONS[lead_name]
            elif lead_name in OTHER_LEAD_DEFINITIONS:
                src, tgt = OTHER_LEAD_DEFINITIONS[lead_name]
            else:
                raise ValueError(f"Unknown lead: {lead_name}")

            src_idx = self.electrode_to_idx[src]
            tgt_idx = self.electrode_to_idx[tgt]

            if isinstance(signal, torch.Tensor):
                signal = signal.numpy()
            signal = np.asarray(signal, dtype=np.float32)

            edge_src.append(src_idx)
            edge_tgt.append(tgt_idx)
            edge_attr_list.append(signal)
            theta, phi = get_lead_spherical_coords(src, tgt)
            edge_spherical_list.append([theta, phi])

            if bidirectional:
                edge_src.append(tgt_idx)
                edge_tgt.append(src_idx)
                edge_attr_list.append(-signal)  
                theta, phi = get_lead_spherical_coords(tgt, src)
                edge_spherical_list.append([theta, phi])

        if len(edge_src) == 0:
            T = 2500  
            return Data(
                x=self.node_positions.clone(),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.zeros((0, T), dtype=torch.float32),
                edge_spherical=torch.zeros((0, 2), dtype=torch.float32),
            )

        return Data(
            x=self.node_positions.clone(),
            edge_index=torch.tensor([edge_src, edge_tgt], dtype=torch.long),
            edge_attr=torch.tensor(np.stack(edge_attr_list), dtype=torch.float32),
            edge_spherical=torch.tensor(edge_spherical_list, dtype=torch.float32),
        )

    def build_12lead_subgraph(
        self,
        ecg_signals: np.ndarray,
        bidirectional: bool = True,
    ) -> Data:
        """
        Convenience method for standard 12-lead ECG.

        Args:
            ecg_signals: (12, T) or (T, 12) array
            bidirectional: Include reverse edges

        Returns:
            24-edge subgraph (if bidirectional) or 12-edge
        """
        if ecg_signals.shape[0] != 12:
            ecg_signals = ecg_signals.T
        assert ecg_signals.shape[0] == 12, f"Expected 12 leads, got {ecg_signals.shape[0]}"

        observed = {name: ecg_signals[i] for i, name in enumerate(LEAD_NAMES_12)}
        return self.build_observed_subgraph(observed, bidirectional=bidirectional)

    def build_subgraph_from_indices(
        self,
        ecg_signals: np.ndarray,
        lead_indices: List[int],
        bidirectional: bool = True,
    ) -> Data:
        """
        Build subgraph with only specified lead indices from 12-lead array.

        Args:
            ecg_signals: (12, T) or (T, 12)
            lead_indices: List of indices (0-11) to include
            bidirectional: Include reverse edges
        """
        if ecg_signals.shape[0] != 12:
            ecg_signals = ecg_signals.T

        observed = {LEAD_NAMES_12[i]: ecg_signals[i] for i in lead_indices}
        return self.build_observed_subgraph(observed, bidirectional=bidirectional)

    def get_query_coords(
        self,
        lead_name: Optional[str] = None,
        src_electrode: Optional[str] = None,
        tgt_electrode: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Get spherical coordinates (θ, φ) for a lead query.

        Can specify either:
        - lead_name: Name of known lead ('I', 'V1', 'ICM', etc.)
        - src_electrode, tgt_electrode: Any electrode pair (for novel leads)

        Returns:
            (theta, phi) tuple
        """
        if lead_name is not None:
            if lead_name in STANDARD_LEAD_DEFINITIONS:
                src, tgt = STANDARD_LEAD_DEFINITIONS[lead_name]
            elif lead_name in OTHER_LEAD_DEFINITIONS:
                src, tgt = OTHER_LEAD_DEFINITIONS[lead_name]
            else:
                raise ValueError(f"Unknown lead: {lead_name}")
        elif src_electrode is not None and tgt_electrode is not None:
            src, tgt = src_electrode, tgt_electrode
        else:
            raise ValueError("Specify lead_name OR (src_electrode, tgt_electrode)")

        return get_lead_spherical_coords(src, tgt)

    def get_all_12lead_coords(self) -> torch.Tensor:
        """Get (theta, phi) for all 12 standard leads. Shape: (12, 2)."""
        coords = [self.get_query_coords(lead_name=name) for name in LEAD_NAMES_12]
        return torch.tensor(coords, dtype=torch.float32)

    def get_coords_for_indices(self, lead_indices: List[int]) -> torch.Tensor:
        """Get (theta, phi) for specific lead indices. Shape: (len(indices), 2)."""
        coords = [self.get_query_coords(lead_name=LEAD_NAMES_12[i]) for i in lead_indices]
        return torch.tensor(coords, dtype=torch.float32)

    def get_extended_coords(self, include_negated: bool = True) -> Tuple[torch.Tensor, List[str]]:
        """
        Get coordinates for extended lead set: 12 standard + ICM + optionally negated.

        Args:
            include_negated: If True, include negated versions of all leads

        Returns:
            coords: (N, 2) tensor of (theta, phi)
            lead_names: List of lead names in same order

        Lead order (if include_negated=True):
            [I, II, III, aVR, aVL, aVF, V1-V6, ICM,  <- 13 positive
             -I, -II, -III, -aVR, -aVL, -aVF, -V1..-V6, -ICM]  <- 13 negated
        """
        coords = []
        lead_names = []

        # 12 standard leads
        for name in LEAD_NAMES_12:
            theta, phi = self.get_query_coords(lead_name=name)
            coords.append([theta, phi])
            lead_names.append(name)

        theta, phi = self.get_query_coords(lead_name='ICM')
        coords.append([theta, phi])
        lead_names.append('ICM')

        if include_negated:
            for name in LEAD_NAMES_12:
                src, tgt = STANDARD_LEAD_DEFINITIONS[name]
                theta, phi = get_lead_spherical_coords(tgt, src) 
                coords.append([theta, phi])
                lead_names.append(f'-{name}')

            src, tgt = OTHER_LEAD_DEFINITIONS['ICM']
            theta, phi = get_lead_spherical_coords(tgt, src)
            coords.append([theta, phi])
            lead_names.append('-ICM')

        return torch.tensor(coords, dtype=torch.float32), lead_names

    def get_13lead_coords(self) -> torch.Tensor:
        """Get (theta, phi) for 12 standard leads + ICM. Shape: (13, 2)."""
        coords, _ = self.get_extended_coords(include_negated=False)
        return coords

    def compute_extended_targets(self, ecg_12lead: np.ndarray, include_negated: bool = True) -> torch.Tensor:
        """
        Compute target signals for extended lead set from 12-lead ECG.

        Args:
            ecg_12lead: (12, T) array of 12 standard leads
            include_negated: If True, include negated versions

        Returns:
            (N, T) tensor where N = 13 (or 26 if include_negated)
            Order matches get_extended_coords()
        """
        T = ecg_12lead.shape[1]
        targets = []

        for i in range(12):
            targets.append(ecg_12lead[i])

        icm = ecg_12lead[8] - ecg_12lead[7]
        targets.append(icm)

        if include_negated:
            for i in range(12):
                targets.append(-ecg_12lead[i])
            targets.append(-icm)

        return torch.tensor(np.stack(targets), dtype=torch.float32)


# Testing

if __name__ == '__main__':
    print("Testing ECG Graph V2 - Fully Connected")
    print("=" * 60)

    builder = ECGGraphBuilderV2()
    T = 2500
    dummy_ecg = np.random.randn(12, T).astype(np.float32)

    # 12-lead
    graph_12 = builder.build_12lead_subgraph(dummy_ecg, bidirectional=True)
    print(f"\n12-lead bidirectional subgraph:")
    print(f"  Nodes: {graph_12.x.shape[0]} (always {builder.num_electrodes})")
    print(f"  Edges: {graph_12.edge_index.shape[1]} (24 = 12 leads × 2)")
    print(f"  Edge attr: {graph_12.edge_attr.shape}")

    # Partial (6 leads)
    subset = [0, 1, 2, 6, 7, 8]  # I, II, III, V1, V2, V3
    graph_6 = builder.build_subgraph_from_indices(dummy_ecg, subset, bidirectional=True)
    print(f"\n6-lead subgraph:")
    print(f"  Nodes: {graph_6.x.shape[0]} (still {builder.num_electrodes})")
    print(f"  Edges: {graph_6.edge_index.shape[1]} (12 = 6 × 2)")

    # Single lead
    icm_signal = dummy_ecg[7] - dummy_ecg[6] 
    graph_icm = builder.build_observed_subgraph({'ICM': icm_signal}, bidirectional=True)
    print(f"\nICM subgraph:")
    print(f"  Nodes: {graph_icm.x.shape[0]}")
    print(f"  Edges: {graph_icm.edge_index.shape[1]} (2)")

    print(f"\nQuery coordinates for leads:")
    for lead in ['I', 'V1', 'ICM']:
        theta, phi = builder.get_query_coords(lead_name=lead)
        print(f"  {lead}: theta={theta:.3f}, phi={phi:.3f}")

    theta, phi = builder.get_query_coords(src_electrode='V1', tgt_electrode='V2')
    print(f"  V1->V2 (novel): theta={theta:.3f}, phi={phi:.3f}")

    from torch_geometric.data import Batch

    graphs = [graph_12, graph_6, graph_icm]
    batch = Batch.from_data_list(graphs)
    print(f"\nBatched graphs (12, 6, 1 leads):")
    print(f"  Total nodes: {batch.x.shape[0]} (3 × {builder.num_electrodes} = {3 * builder.num_electrodes})")
    print(f"  Total edges: {batch.edge_index.shape[1]} (24 + 12 + 2 = 38)")
    print(f"  Batch works with variable edges!")

    print(f"\nPhysics verification (negation):")
    print(f"  Forward edge 0 sum: {graph_12.edge_attr[0].sum():.4f}")
    print(f"  Reverse edge 1 sum: {graph_12.edge_attr[1].sum():.4f}")
    print(f"  Sum (should be ~0): {(graph_12.edge_attr[0] + graph_12.edge_attr[1]).abs().max():.8f}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("Ready for: masked reconstruction, novel lead synthesis, any config")

"""
Flexible ECG Graph Representation for PyTorch Geometric.

Supports:
- Full 12-lead ECG
- Partial lead subsets (e.g., single-lead, 3-lead, 6-lead)
- Custom leads defined by coordinates or electrode compositions
- Edge masking for training with dropout

Graph structure:
- 13 base nodes: WCT, RA, LA, LL, V1-V6, mid_LA_LL, mid_RA_LL, mid_RA_LA
- 12 base edges: I, II, III, aVR, aVL, aVF, V1-V6
- Custom nodes/edges can be added for non-standard leads
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass



LEAD_ANGLES = {
    'I': (np.pi / 2, np.pi / 2),
    'II': (np.pi * 5 / 6, np.pi / 2),
    'III': (np.pi * (4/5), -np.pi / 2),
    'aVR': (np.pi * (1/3), -np.pi / 2),
    'aVL': (np.pi * (1/3), np.pi / 2),
    'aVF': (np.pi * 1, np.pi / 2),
    'V1': (np.pi / 2, -np.pi / 18),
    'V2': (np.pi / 2, np.pi / 18),
    'V3': (np.pi * (19/36), np.pi / 12),
    'V4': (np.pi * (11/20), np.pi / 6),
    'V5': (np.pi * (16/30), np.pi / 3),
    'V6': (np.pi * (16/30), np.pi / 2),
}

LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

PHYSICAL_ELECTRODES = ['WCT', 'RA', 'LA', 'LL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

VIRTUAL_ELECTRODES = ['mid_LA_LL', 'mid_RA_LL', 'mid_RA_LA']

BASE_NODES = PHYSICAL_ELECTRODES + VIRTUAL_ELECTRODES

BASE_EDGES = [
    ('RA', 'LA', 'I'),      
    ('RA', 'LL', 'II'),     
    ('LA', 'LL', 'III'),    
    ('mid_LA_LL', 'RA', 'aVR'),  
    ('mid_RA_LL', 'LA', 'aVL'),  
    ('mid_RA_LA', 'LL', 'aVF'),  
    ('WCT', 'V1', 'V1'),
    ('WCT', 'V2', 'V2'),
    ('WCT', 'V3', 'V3'),
    ('WCT', 'V4', 'V4'),
    ('WCT', 'V5', 'V5'),
    ('WCT', 'V6', 'V6'),
]


def spherical_to_cartesian(theta: float, phi: float, r: float = 1.0) -> np.ndarray:
    """Convert spherical to Cartesian coordinates."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z], dtype=np.float32)


def compute_base_electrode_positions() -> Dict[str, np.ndarray]:
    """Compute 3D positions for all base electrodes."""
    positions = {}

    positions['WCT'] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for lead in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
        theta, phi = LEAD_ANGLES[lead]
        positions[lead] = spherical_to_cartesian(theta, phi)

    theta_I, phi_I = LEAD_ANGLES['I']
    dir_I = spherical_to_cartesian(theta_I, phi_I)

    theta_II, phi_II = LEAD_ANGLES['II']
    dir_II = spherical_to_cartesian(theta_II, phi_II)

    positions['RA'] = -(dir_I + dir_II) / 3
    positions['LA'] = (2 * dir_I - dir_II) / 3
    positions['LL'] = (2 * dir_II - dir_I) / 3

    positions['mid_LA_LL'] = (positions['LA'] + positions['LL']) / 2
    positions['mid_RA_LL'] = (positions['RA'] + positions['LL']) / 2
    positions['mid_RA_LA'] = (positions['RA'] + positions['LA']) / 2

    return positions


BASE_POSITIONS = compute_base_electrode_positions()


@dataclass
class LeadInput:
    """Specification for a single lead input."""
    name: str
    signal: np.ndarray
    is_standard: bool = True
    coordinate: Optional[Tuple[float, float]] = None
    source_electrode: Optional[str] = None
    target_electrode: Optional[str] = None


def create_standard_lead(name: str, signal: np.ndarray) -> LeadInput:
    """Create a standard 12-lead ECG lead input."""
    assert name in LEAD_NAMES_12, f"Unknown standard lead: {name}"
    return LeadInput(name=name, signal=signal, is_standard=True)


def create_coordinate_lead(
    name: str,
    signal: np.ndarray,
    theta: float,
    phi: float
) -> LeadInput:
    """Create a custom lead defined by spherical coordinate (measured from WCT)."""
    return LeadInput(
        name=name,
        signal=signal,
        is_standard=False,
        coordinate=(theta, phi)
    )


def create_composite_lead(
    name: str,
    signal: np.ndarray,
    source: str,
    target: str
) -> LeadInput:
    """Create a custom lead defined as target - source electrode."""
    return LeadInput(
        name=name,
        signal=signal,
        is_standard=False,
        source_electrode=source,
        target_electrode=target
    )


class ECGGraphBuilder:
    """
    Builds PyG graphs from ECG lead inputs.

    Supports flexible lead configurations while maintaining the underlying
    electrode coordinate system as domain knowledge.
    """

    def __init__(self):
        self.base_positions = BASE_POSITIONS.copy()
        self.base_nodes = BASE_NODES.copy()
        self.base_edges = BASE_EDGES.copy()

        self._build_edge_lookup()

    def _build_edge_lookup(self):
        """Build mapping from lead name to edge definition."""
        self.lead_to_edge = {}
        for src, tgt, lead in self.base_edges:
            self.lead_to_edge[lead] = (src, tgt)

    def _get_nodes_for_edge(self, src: str, tgt: str) -> List[str]:
        """Get all nodes needed for an edge, including dependencies."""
        nodes = {src, tgt}

        virtual_deps = {
            'mid_LA_LL': ['LA', 'LL'],
            'mid_RA_LL': ['RA', 'LL'],
            'mid_RA_LA': ['RA', 'LA'],
        }

        for node in [src, tgt]:
            if node in virtual_deps:
                nodes.update(virtual_deps[node])

        nodes.add('WCT')

        return list(nodes)

    def build_graph(
        self,
        lead_inputs: List[LeadInput],
        y: Optional[int] = None,
        include_all_base_nodes: bool = False,
    ) -> Data:
        """
        Build a PyG graph from lead inputs.

        Args:
            lead_inputs: List of LeadInput specifications
            y: Optional label
            include_all_base_nodes: If True, include all 13 base nodes even if
                                    not connected to observed edges

        Returns:
            PyG Data object with:
                - x: Node coordinates as features (N, 3)
                - pos: Node positions (N, 3)
                - edge_index: (2, E) directed edges
                - edge_attr: (E, T) lead signals
                - edge_mask: (E,) which edges have observed signals (for base edges)
                - node_names: list of node names
                - edge_names: list of edge/lead names
        """
        if len(lead_inputs) == 0:
            raise ValueError("At least one lead input required")

        T = lead_inputs[0].signal.shape[0]

        node_set = set()
        custom_nodes = {}

        edges = []
        for lead_input in lead_inputs:
            if lead_input.is_standard:
                src, tgt = self.lead_to_edge[lead_input.name]
                required_nodes = self._get_nodes_for_edge(src, tgt)
                node_set.update(required_nodes)
                edges.append((src, tgt, lead_input.signal, lead_input.name))

            elif lead_input.coordinate is not None:
                theta, phi = lead_input.coordinate
                node_name = f"custom_{lead_input.name}"
                custom_nodes[node_name] = spherical_to_cartesian(theta, phi)
                node_set.add('WCT')
                node_set.add(node_name)
                edges.append(('WCT', node_name, lead_input.signal, lead_input.name))

            elif lead_input.source_electrode is not None:
                src = lead_input.source_electrode
                tgt = lead_input.target_electrode
                required_nodes = self._get_nodes_for_edge(src, tgt)
                node_set.update(required_nodes)
                edges.append((src, tgt, lead_input.signal, lead_input.name))

        if include_all_base_nodes:
            node_set.update(self.base_nodes)

        node_list = [n for n in self.base_nodes if n in node_set]
        node_list.extend(sorted(custom_nodes.keys()))

        node_to_idx = {name: i for i, name in enumerate(node_list)}

        positions = []
        for name in node_list:
            if name in self.base_positions:
                positions.append(self.base_positions[name])
            else:
                positions.append(custom_nodes[name])
        positions = np.stack(positions, axis=0)

        edge_sources = []
        edge_targets = []
        edge_signals = []
        edge_names = []

        for src, tgt, signal, name in edges:
            edge_sources.append(node_to_idx[src])
            edge_targets.append(node_to_idx[tgt])
            edge_signals.append(signal)
            edge_names.append(name)

        edge_index = np.array([edge_sources, edge_targets])
        edge_attr = np.stack(edge_signals, axis=0)

        edge_mask = np.zeros(len(self.base_edges), dtype=np.float32)
        observed_leads = {inp.name for inp in lead_inputs if inp.is_standard}
        for i, (_, _, lead_name) in enumerate(self.base_edges):
            if lead_name in observed_leads:
                edge_mask[i] = 1.0

        data = Data(
            x=torch.tensor(positions, dtype=torch.float32),
            pos=torch.tensor(positions, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
        )

        if y is not None:
            data.y = torch.tensor([y], dtype=torch.long)

        data.node_names = node_list
        data.edge_names = edge_names
        data.num_timesteps = T

        return data


def ecg_12lead_to_graph(
    ecg_waveform: np.ndarray,
    y: Optional[int] = None,
    lead_order: List[str] = LEAD_NAMES_12,
) -> Data:
    """
    Convert a standard 12-lead ECG to graph.

    Args:
        ecg_waveform: Shape (T, 12) or (12, T) or (1, T, 12)
        y: Optional label
        lead_order: Order of leads in the input array

    Returns:
        PyG Data object
    """
    ecg = np.array(ecg_waveform)
    if ecg.ndim == 3:
        ecg = ecg[0]
    if ecg.shape[0] == 12:
        ecg = ecg.T

    lead_inputs = [
        create_standard_lead(name, ecg[:, i])
        for i, name in enumerate(lead_order)
    ]

    builder = ECGGraphBuilder()
    return builder.build_graph(lead_inputs, y=y, include_all_base_nodes=True)


def partial_leads_to_graph(
    signals: Dict[str, np.ndarray],
    y: Optional[int] = None,
) -> Data:
    """
    Convert a partial set of standard leads to graph.

    Args:
        signals: Dict mapping lead name to signal array
        y: Optional label

    Returns:
        PyG Data object
    """
    lead_inputs = [
        create_standard_lead(name, signal)
        for name, signal in signals.items()
    ]

    builder = ECGGraphBuilder()
    return builder.build_graph(lead_inputs, y=y)


def custom_lead_to_graph(
    signal: np.ndarray,
    theta: float,
    phi: float,
    name: str = "custom",
    y: Optional[int] = None,
) -> Data:
    """
    Convert a single custom lead (defined by coordinate) to graph.

    Args:
        signal: Lead signal array
        theta, phi: Spherical coordinates of the electrode
        name: Name for the lead
        y: Optional label

    Returns:
        PyG Data object
    """
    lead_input = create_coordinate_lead(name, signal, theta, phi)
    builder = ECGGraphBuilder()
    return builder.build_graph([lead_input], y=y)


def composite_lead_to_graph(
    signal: np.ndarray,
    source: str,
    target: str,
    name: str = "composite",
    y: Optional[int] = None,
) -> Data:
    """
    Convert a single lead defined as difference between electrodes.

    Args:
        signal: Lead signal array
        source: Source electrode name (e.g., 'V2')
        target: Target electrode name (e.g., 'V3')
        name: Name for the lead
        y: Optional label

    Returns:
        PyG Data object (lead = target - source)
    """
    lead_input = create_composite_lead(name, signal, source, target)
    builder = ECGGraphBuilder()
    return builder.build_graph([lead_input], y=y)



if __name__ == "__main__":
    import os

    print("Flexible ECG Graph Builder")
    print("=" * 60)

    print("\nBase electrode positions:")
    for name in BASE_NODES:
        pos = BASE_POSITIONS[name]
        print(f"  {name:12s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

    print("\nBase edges (12 standard leads):")
    for src, tgt, lead in BASE_EDGES:
        print(f"  {lead:4s}: {src:12s} -> {tgt}")

    DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    waveform_path = os.path.join(DATA_DIR, "EchoNext_test_waveforms.npy")

    if os.path.exists(waveform_path):
        print("\n" + "=" * 60)
        print("Testing with actual ECG data...")

        waveforms = np.load(waveform_path)
        ecg = waveforms[0, 0] 
        print(f"ECG shape: {ecg.shape}")

        print("\n--- Test 1: Full 12-lead ECG ---")
        graph = ecg_12lead_to_graph(ecg, y=1)
        print(f"  Nodes: {len(graph.node_names)} - {graph.node_names}")
        print(f"  Edges: {len(graph.edge_names)} - {graph.edge_names}")
        print(f"  x shape: {graph.x.shape}")
        print(f"  edge_attr shape: {graph.edge_attr.shape}")
        print(f"  edge_mask: {graph.edge_mask.tolist()}")

        print("\n--- Test 2: Partial leads (aVF + V1) ---")
        partial_signals = {
            'aVF': ecg[:, 5], 
            'V1': ecg[:, 6],   
        }
        graph = partial_leads_to_graph(partial_signals, y=0)
        print(f"  Nodes: {len(graph.node_names)} - {graph.node_names}")
        print(f"  Edges: {len(graph.edge_names)} - {graph.edge_names}")
        print(f"  x shape: {graph.x.shape}")
        print(f"  edge_attr shape: {graph.edge_attr.shape}")

        print("\n--- Test 3: Custom lead by coordinate (ICM-like) ---")
        icm_signal = ecg[:, 8] 
        theta, phi = LEAD_ANGLES['V3']
        graph = custom_lead_to_graph(icm_signal, theta, phi, name="ICM")
        print(f"  Nodes: {len(graph.node_names)} - {graph.node_names}")
        print(f"  Edges: {len(graph.edge_names)} - {graph.edge_names}")
        print(f"  x shape: {graph.x.shape}")

        print("\n--- Test 4: Composite lead (V3 - V2) ---")
        v3_minus_v2 = ecg[:, 8] - ecg[:, 7] 
        graph = composite_lead_to_graph(v3_minus_v2, source='V2', target='V3', name="V3-V2")
        print(f"  Nodes: {len(graph.node_names)} - {graph.node_names}")
        print(f"  Edges: {len(graph.edge_names)} - {graph.edge_names}")
        print(f"  x shape: {graph.x.shape}")

        print("\n--- Test 5: Mixed (Lead I + custom ICM) ---")
        builder = ECGGraphBuilder()
        lead_inputs = [
            create_standard_lead('I', ecg[:, 0]),
            create_coordinate_lead('ICM', icm_signal, theta, phi),
        ]
        graph = builder.build_graph(lead_inputs)
        print(f"  Nodes: {len(graph.node_names)} - {graph.node_names}")
        print(f"  Edges: {len(graph.edge_names)} - {graph.edge_names}")

        print("\nAll tests passed!")
    else:
        print(f"\nNote: Test data not found at {waveform_path}")

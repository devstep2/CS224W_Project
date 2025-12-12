"""
Create figures for ECG Graph V2 architecture.

Generates:
1. Full 13-electrode graph with all 156 possible edges (supergraph)
2. Standard 12-lead subgraph (24 bidirectional edges)
3. 6-limb-lead subgraph (12 bidirectional edges) - training input example
4. Comparison panel showing masking/dropout strategy
5. Spherical coordinate visualization for lead synthesis
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import os

from ecg_graph_flexible import BASE_POSITIONS, BASE_EDGES, BASE_NODES, LEAD_NAMES_12

ALL_ELECTRODES = BASE_NODES
ELECTRODE_POSITIONS = BASE_POSITIONS
STANDARD_LEAD_DEFINITIONS = {
    'I':   ('RA', 'LA'),
    'II':  ('RA', 'LL'),
    'III': ('LA', 'LL'),
    'aVR': ('mid_LA_LL', 'RA'),
    'aVL': ('mid_RA_LL', 'LA'),
    'aVF': ('mid_RA_LA', 'LL'),
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


def get_lead_spherical_coords(src_electrode: str, tgt_electrode: str):
    """Get spherical coordinates for a lead given electrode names."""
    src_pos = ELECTRODE_POSITIONS[src_electrode]
    tgt_pos = ELECTRODE_POSITIONS[tgt_electrode]
    direction = tgt_pos - src_pos
    norm = np.linalg.norm(direction)
    if norm > 1e-8:
        direction = direction / norm
    x, y, z = direction
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    theta = np.arctan2(y, x)
    if theta < 0:
        theta += 2 * np.pi
    return theta, phi


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['mathtext.fontset'] = 'stix'

ELECTRODE_COLORS = {
    'WCT': '#27ae60',
    'RA': '#e74c3c',
    'LA': '#e74c3c',
    'LL': '#e74c3c',
    'V1': '#3498db',
    'V2': '#3498db',
    'V3': '#3498db',
    'V4': '#3498db',
    'V5': '#3498db',
    'V6': '#3498db',
    'mid_LA_LL': '#9b59b6',
    'mid_RA_LL': '#9b59b6',
    'mid_RA_LA': '#9b59b6',
}

LEAD_COLORS = {
    'limb': '#c0392b',
    'augmented': '#8e44ad',
    'precordial': '#2980b9',
    'icm': '#16a085',
    'background': '#bdc3c7',
}


def get_electrode_type(name):
    """Classify electrode by type."""
    if name == 'WCT':
        return 'reference'
    elif name in ['RA', 'LA', 'LL']:
        return 'limb'
    elif name.startswith('V'):
        return 'precordial'
    else:
        return 'virtual'


def get_lead_type(lead_name):
    """Classify lead by type."""
    if lead_name in ['I', 'II', 'III']:
        return 'limb'
    elif lead_name in ['aVR', 'aVL', 'aVF']:
        return 'augmented'
    elif lead_name.startswith('V'):
        return 'precordial'
    elif lead_name == 'ICM':
        return 'icm'
    return 'background'


def project_to_2d(pos3d, view='frontal'):
    """Project 3D position to 2D based on viewing plane."""
    x, y, z = pos3d
    if view == 'frontal':
        return (y, z)
    elif view == 'transverse':
        return (x, y)
    elif view == 'sagittal':
        return (x, z)
    return (y, z)


def draw_curved_arrow(ax, start, end, color='black', alpha=0.7, lw=1.5,
                      curve=0.15, head_width=0.04, head_length=0.03):
    """Draw a curved arrow between two points in 2D."""
    mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = np.sqrt(dx**2 + dy**2)
    if length < 0.01:
        return

    px, py = -dy / length * curve, dx / length * curve
    ctrl = (mid[0] + px, mid[1] + py)

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=alpha,
                               connectionstyle=f'arc3,rad={curve}'))


def create_full_connected_figure():
    """
    Create figure showing the fully connected 13-electrode supergraph
    with 156 possible directed edges.
    """
    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('A. 13-Electrode Supergraph\n(156 possible edges)', fontweight='bold', fontsize=11)

    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]
        color = ELECTRODE_COLORS[name]
        size = 200 if name == 'WCT' else (120 if 'mid' in name else 150)
        marker = 's' if 'mid' in name else 'o'
        ax1.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=0.5, zorder=5)

        label = name if name in ['WCT', 'RA', 'LA', 'LL'] or name.startswith('V') else ''
        if label:
            ax1.text(pos[0] + 0.1, pos[1] + 0.1, pos[2] + 0.05, label, fontsize=8)

    for i, src in enumerate(ALL_ELECTRODES):
        for j, tgt in enumerate(ALL_ELECTRODES):
            if i < j:
                src_pos = ELECTRODE_POSITIONS[src]
                tgt_pos = ELECTRODE_POSITIONS[tgt]
                ax1.plot([src_pos[0], tgt_pos[0]],
                        [src_pos[1], tgt_pos[1]],
                        [src_pos[2], tgt_pos[2]],
                        color='#bdc3c7', alpha=0.15, linewidth=0.5)

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1.2, 0.5)
    ax1.set_ylim(-1.4, 0.6)
    ax1.set_zlim(-0.5, 0.7)
    ax1.view_init(elev=20, azim=135)

    ax2 = fig.add_subplot(132)
    ax2.set_title('B. Standard 12-Lead Subgraph\n(24 bidirectional edges)', fontweight='bold', fontsize=11)
    ax2.set_aspect('equal')
    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]
        y, z = project_to_2d(pos, 'frontal')
        color = ELECTRODE_COLORS[name]
        size = 150 if name == 'WCT' else (100 if 'mid' in name else 120)
        marker = 's' if 'mid' in name else 'o'
        ax2.scatter(y, z, c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=0.8, zorder=5)

        if name in ['WCT', 'RA', 'LA', 'LL'] or name.startswith('V'):
            offset = (5, 5) if name not in ['RA'] else (-25, 5)
            ax2.annotate(name, (y, z), xytext=offset, textcoords='offset points', fontsize=7)
    for lead_name in LEAD_NAMES_12:
        src, tgt = STANDARD_LEAD_DEFINITIONS[lead_name]
        src_pos = project_to_2d(ELECTRODE_POSITIONS[src], 'frontal')
        tgt_pos = project_to_2d(ELECTRODE_POSITIONS[tgt], 'frontal')

        lead_type = get_lead_type(lead_name)
        color = LEAD_COLORS[lead_type]

        draw_curved_arrow(ax2, src_pos, tgt_pos, color=color, alpha=0.8, lw=1.5, curve=0.1)
        draw_curved_arrow(ax2, tgt_pos, src_pos, color=color, alpha=0.5, lw=1.0, curve=-0.1)

    ax2.set_xlabel('Y (Left-Right)')
    ax2.set_ylabel('Z (Superior-Inferior)')
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.0, 0.6)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.2)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.2)

    ax3 = fig.add_subplot(133)
    ax3.set_title('C. 6-Limb-Lead Subgraph\n(12 bidirectional edges)', fontweight='bold', fontsize=11)
    ax3.set_aspect('equal')

    limb_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']
    limb_electrodes = ['WCT', 'RA', 'LA', 'LL', 'mid_LA_LL', 'mid_RA_LL', 'mid_RA_LA']

    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]
        y, z = project_to_2d(pos, 'frontal')

        if name in limb_electrodes:
            color = ELECTRODE_COLORS[name]
            size = 150 if name == 'WCT' else (100 if 'mid' in name else 120)
            alpha = 1.0
        else:
            color = '#e0e0e0'
            size = 60
            alpha = 0.4

        marker = 's' if 'mid' in name else 'o'
        ax3.scatter(y, z, c=color, s=size, marker=marker,
                   edgecolors='gray', linewidths=0.5, zorder=5, alpha=alpha)

        if name in limb_electrodes and name in ['RA', 'LA', 'LL']:
            offset = (5, 5) if name not in ['RA'] else (-25, 5)
            ax3.annotate(name, (y, z), xytext=offset, textcoords='offset points', fontsize=8)
    for lead_name in limb_leads:
        src, tgt = STANDARD_LEAD_DEFINITIONS[lead_name]
        src_pos = project_to_2d(ELECTRODE_POSITIONS[src], 'frontal')
        tgt_pos = project_to_2d(ELECTRODE_POSITIONS[tgt], 'frontal')

        lead_type = get_lead_type(lead_name)
        color = LEAD_COLORS[lead_type]

        draw_curved_arrow(ax3, src_pos, tgt_pos, color=color, alpha=0.9, lw=2, curve=0.1)
        draw_curved_arrow(ax3, tgt_pos, src_pos, color=color, alpha=0.6, lw=1.2, curve=-0.1)

    ax3.set_xlabel('Y (Left-Right)')
    ax3.set_ylabel('Z (Superior-Inferior)')
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.0, 0.6)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.2)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.2)

    ax3.text(0.5, 0.05, 'Precordial leads\n(masked)', transform=ax3.transAxes,
            fontsize=9, ha='center', va='bottom', style='italic', color='gray')

    legend_elements = [
        mpatches.Patch(color='#27ae60', label='WCT (Reference)'),
        mpatches.Patch(color='#e74c3c', label='Limb Electrodes (RA, LA, LL)'),
        mpatches.Patch(color='#3498db', label='Precordial (V1-V6)'),
        mpatches.Patch(color='#9b59b6', label='Virtual Midpoints'),
        Line2D([0], [0], color=LEAD_COLORS['limb'], lw=2, label='Limb Leads (I, II, III)'),
        Line2D([0], [0], color=LEAD_COLORS['augmented'], lw=2, label='Augmented (aVR, aVL, aVF)'),
        Line2D([0], [0], color=LEAD_COLORS['precordial'], lw=2, label='Precordial (V1-V6)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.05), fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig


def create_bidirectional_physics_figure():
    """
    Create figure explaining bidirectional edge physics:
    Forward edge carries signal, reverse edge carries NEGATED signal.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax1 = axes[0]
    ax1.set_title('A. Lead as Voltage Difference', fontweight='bold', fontsize=11)
    ax1.set_aspect('equal')
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(-0.5, 1.5)

    ra_pos = (0.3, 1.0)
    la_pos = (1.7, 1.0)
    ax1.scatter(*ra_pos, s=300, c='#e74c3c', edgecolors='black', linewidths=2, zorder=5)
    ax1.scatter(*la_pos, s=300, c='#e74c3c', edgecolors='black', linewidths=2, zorder=5)
    ax1.text(ra_pos[0], ra_pos[1] + 0.2, 'RA', ha='center', fontsize=12, fontweight='bold')
    ax1.text(la_pos[0], la_pos[1] + 0.2, 'LA', ha='center', fontsize=12, fontweight='bold')

    ax1.annotate('', xy=(1.4, 0.9), xytext=(0.6, 0.9),
                arrowprops=dict(arrowstyle='->', color=LEAD_COLORS['limb'], lw=3))
    ax1.text(1.0, 0.65, r'Lead I = $V_{LA} - V_{RA}$', ha='center', fontsize=11,
            color=LEAD_COLORS['limb'], fontweight='bold')

    ax1.annotate('', xy=(0.6, 1.1), xytext=(1.4, 1.1),
                arrowprops=dict(arrowstyle='->', color=LEAD_COLORS['limb'], lw=2, alpha=0.6,
                               linestyle='--'))
    ax1.text(1.0, 1.25, r'$-(V_{LA} - V_{RA}) = V_{RA} - V_{LA}$', ha='center', fontsize=9,
            color='gray', style='italic')

    ax1.text(1.0, 0.15, 'Bidirectional: reverse = negated signal', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

    ax1.axis('off')

    ax2 = axes[1]
    ax2.set_title('B. Signal Negation Property', fontweight='bold', fontsize=11)

    t = np.linspace(0, 1, 500)
    signal = 0.5 * np.sin(2 * np.pi * 3 * t) * np.exp(-3 * (t - 0.3)**2) + \
             0.3 * np.sin(2 * np.pi * 8 * t) * np.exp(-5 * (t - 0.5)**2)

    ax2.plot(t, signal, color=LEAD_COLORS['limb'], lw=2, label='Forward: RA→LA (+signal)')
    ax2.plot(t, -signal, color=LEAD_COLORS['limb'], lw=2, alpha=0.6, linestyle='--',
            label='Reverse: LA→RA (−signal)')
    ax2.axhline(0, color='gray', lw=0.5, alpha=0.5)

    ax2.set_xlabel('Time (normalized)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.set_title('C. Subgraph from Lead Selection', fontweight='bold', fontsize=11)
    ax3.set_aspect('equal')

    electrodes_subset = ['RA', 'LA', 'LL', 'V1', 'V2']
    positions_2d = {
        'RA': (-0.8, 0.5), 'LA': (0.8, 0.5), 'LL': (0.0, -0.8),
        'V1': (0.3, 0.0), 'V2': (-0.3, 0.0), 'WCT': (0.0, -0.2)
    }

    # Draw potential edges faintly
    all_nodes = list(positions_2d.keys())
    for i, n1 in enumerate(all_nodes):
        for j, n2 in enumerate(all_nodes):
            if i < j:
                p1, p2 = positions_2d[n1], positions_2d[n2]
                ax3.plot([p1[0], p2[0]], [p1[1], p2[1]],
                        color='#e0e0e0', lw=0.5, alpha=0.5)

    # Draw selected leads strongly
    selected = [('RA', 'LA', 'I'), ('RA', 'LL', 'II')]
    for src, tgt, name in selected:
        p1, p2 = positions_2d[src], positions_2d[tgt]
        ax3.annotate('', xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle='->', color=LEAD_COLORS['limb'], lw=2.5))
        mid = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
        ax3.text(mid[0] + 0.1, mid[1] + 0.1, name, fontsize=10, fontweight='bold',
                color=LEAD_COLORS['limb'])

    # Draw nodes
    for name, pos in positions_2d.items():
        if name in electrodes_subset:
            color = ELECTRODE_COLORS.get(name, '#95a5a6')
            size = 200
            alpha = 1.0
        else:
            color = '#e0e0e0'
            size = 100
            alpha = 0.5
        ax3.scatter(*pos, s=size, c=color, edgecolors='black', linewidths=1,
                   zorder=5, alpha=alpha)
        if name in electrodes_subset:
            ax3.text(pos[0], pos[1] - 0.15, name, ha='center', fontsize=9)

    ax3.text(0.0, -1.15, 'Selected leads → Subgraph\n(rest of supergraph masked)',
            ha='center', fontsize=9, style='italic')
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 0.9)
    ax3.axis('off')

    plt.tight_layout()
    return fig


def create_3d_electrode_graph_figure():
    """
    Create a standalone 3D figure showing the 13-electrode graph with all electrodes
    as nodes and the 12 standard leads as edges.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('ECG Graph: 13 Electrode Nodes with 12-Lead Edges', fontweight='bold', fontsize=14)

    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]
        color = ELECTRODE_COLORS[name]
        size = 300 if name == 'WCT' else (180 if 'mid' in name else 220)
        marker = 's' if 'mid' in name else 'o'
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=1.5, zorder=5)

        label = name.replace('mid_', 'm').replace('_', '-')
        offset_x, offset_y, offset_z = 0.08, 0.08, 0.05
        if name == 'WCT':
            offset_x, offset_y, offset_z = -0.15, -0.1, 0.05
        elif name.startswith('V'):
            offset_x = 0.1
        ax.text(pos[0] + offset_x, pos[1] + offset_y, pos[2] + offset_z,
                name, fontsize=10, fontweight='bold')

    edge_colors_map = {
        'I': '#c0392b', 'II': '#c0392b', 'III': '#c0392b',
        'aVR': '#8e44ad', 'aVL': '#8e44ad', 'aVF': '#8e44ad',
        'V1': '#2980b9', 'V2': '#2980b9', 'V3': '#2980b9',
        'V4': '#2980b9', 'V5': '#2980b9', 'V6': '#2980b9',
    }

    for lead_name, (src, tgt) in STANDARD_LEAD_DEFINITIONS.items():
        src_pos = ELECTRODE_POSITIONS[src]
        tgt_pos = ELECTRODE_POSITIONS[tgt]
        color = edge_colors_map[lead_name]

        ax.plot([src_pos[0], tgt_pos[0]],
                [src_pos[1], tgt_pos[1]],
                [src_pos[2], tgt_pos[2]],
                color=color, alpha=0.8, linewidth=2.5)

        ax.scatter([tgt_pos[0]], [tgt_pos[1]], [tgt_pos[2]],
                   color=color, s=40, marker='>', alpha=0.9, zorder=4)

        mid = (np.array(src_pos) + np.array(tgt_pos)) / 2
        ax.text(mid[0], mid[1], mid[2], lead_name, fontsize=8,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7))

    ax.set_xlabel('X (Anterior)', fontsize=11)
    ax.set_ylabel('Y (Left)', fontsize=11)
    ax.set_zlabel('Z (Superior)', fontsize=11)
    all_pos = np.array([ELECTRODE_POSITIONS[n] for n in ALL_ELECTRODES])
    margin = 0.3
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    ax.view_init(elev=25, azim=45)

    legend_elements = [
        mpatches.Patch(color='#27ae60', label='WCT (Reference)'),
        mpatches.Patch(color='#e74c3c', label='Limb Electrodes (RA, LA, LL)'),
        mpatches.Patch(color='#3498db', label='Precordial (V1-V6)'),
        mpatches.Patch(color='#9b59b6', label='Virtual Midpoints'),
        Line2D([0], [0], color='#c0392b', lw=2.5, label='Limb Leads (I, II, III)'),
        Line2D([0], [0], color='#8e44ad', lw=2.5, label='Augmented (aVR, aVL, aVF)'),
        Line2D([0], [0], color='#2980b9', lw=2.5, label='Precordial (V1-V6)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.tight_layout()
    return fig


def create_fully_connected_3d_figure():
    """
    Create a standalone 3D figure showing the fully connected 13-electrode supergraph
    (all 156 directed edges).
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Fully Connected 13-Electrode Supergraph\n(156 directed edges)',
                 fontweight='bold', fontsize=14)

    electrodes_list = list(ALL_ELECTRODES)
    for i, src in enumerate(electrodes_list):
        for j, tgt in enumerate(electrodes_list):
            if i < j:
                src_pos = ELECTRODE_POSITIONS[src]
                tgt_pos = ELECTRODE_POSITIONS[tgt]
                ax.plot([src_pos[0], tgt_pos[0]],
                        [src_pos[1], tgt_pos[1]],
                        [src_pos[2], tgt_pos[2]],
                        color='#7f8c8d', alpha=0.45, linewidth=1.2)

    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]
        color = ELECTRODE_COLORS[name]
        size = 300 if name == 'WCT' else (180 if 'mid' in name else 220)
        marker = 's' if 'mid' in name else 'o'
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=1.5, zorder=5)

        label = name.replace('mid_', 'm').replace('_', '-')
        offset_x, offset_y, offset_z = 0.08, 0.08, 0.05
        if name == 'WCT':
            offset_x, offset_y, offset_z = -0.15, -0.1, 0.05
        elif name.startswith('V'):
            offset_x = 0.1
        ax.text(pos[0] + offset_x, pos[1] + offset_y, pos[2] + offset_z,
                name, fontsize=10, fontweight='bold')

    ax.set_xlabel('X (Anterior)', fontsize=11)
    ax.set_ylabel('Y (Left)', fontsize=11)
    ax.set_zlabel('Z (Superior)', fontsize=11)
    all_pos = np.array([ELECTRODE_POSITIONS[n] for n in ALL_ELECTRODES])
    margin = 0.3
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    ax.view_init(elev=25, azim=45)

    legend_elements = [
        mpatches.Patch(color='#27ae60', label='WCT (Reference)'),
        mpatches.Patch(color='#e74c3c', label='Limb Electrodes (RA, LA, LL)'),
        mpatches.Patch(color='#3498db', label='Precordial (V1-V6)'),
        mpatches.Patch(color='#9b59b6', label='Virtual Midpoints'),
        Line2D([0], [0], color='#7f8c8d', lw=2, alpha=0.6, label='Potential Edges (156 directed)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    # Add note about edge count
    ax.text2D(0.02, 0.02, 'n(n-1) = 13×12 = 156 directed edges\n'
              '(each pair has forward + reverse)',
              transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_supergraph_with_icm_highlight():
    """
    Create a 3D figure showing the fully connected supergraph with the ICM lead
    (V2 → V3) highlighted as an example of selecting a single edge.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Supergraph with ICM Lead Highlighted\n(Example: selecting V2 → V3 edge)',
                 fontweight='bold', fontsize=14)

    electrodes_list = list(ALL_ELECTRODES)
    for i, src in enumerate(electrodes_list):
        for j, tgt in enumerate(electrodes_list):
            if i < j:
                src_pos = ELECTRODE_POSITIONS[src]
                tgt_pos = ELECTRODE_POSITIONS[tgt]
                ax.plot([src_pos[0], tgt_pos[0]],
                        [src_pos[1], tgt_pos[1]],
                        [src_pos[2], tgt_pos[2]],
                        color='#bdc3c7', alpha=0.3, linewidth=0.8)

    v2_pos = ELECTRODE_POSITIONS['V2']
    v3_pos = ELECTRODE_POSITIONS['V3']

    ax.plot([v2_pos[0], v3_pos[0]],
            [v2_pos[1], v3_pos[1]],
            [v2_pos[2], v3_pos[2]],
            color='#e74c3c', alpha=1.0, linewidth=4, zorder=10)

    ax.scatter([v3_pos[0]], [v3_pos[1]], [v3_pos[2]],
               color='#e74c3c', s=80, marker='>', alpha=1.0, zorder=11)

    mid_icm = (np.array(v2_pos) + np.array(v3_pos)) / 2
    ax.text(mid_icm[0] - 0.15, mid_icm[1] + 0.2, mid_icm[2] + 0.15, 'ICM', fontsize=14,
            fontweight='bold', color='#c0392b', zorder=20,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.95))

    for name in ALL_ELECTRODES:
        pos = ELECTRODE_POSITIONS[name]

        if name in ['V2', 'V3']:
            color = '#e74c3c'
            size = 350
            edgewidth = 2.5
        else:
            color = ELECTRODE_COLORS[name]
            size = 250 if name == 'WCT' else (150 if 'mid' in name else 180)
            edgewidth = 1.0

        marker = 's' if 'mid' in name else 'o'
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, marker=marker,
                   edgecolors='black', linewidths=edgewidth, zorder=5)

        offset_x, offset_y, offset_z = 0.08, 0.08, 0.05
        if name == 'WCT':
            offset_x, offset_y, offset_z = -0.15, -0.1, 0.05
        elif name.startswith('V'):
            offset_x = 0.1

        fontweight = 'bold' if name in ['V2', 'V3'] else 'normal'
        ax.text(pos[0] + offset_x, pos[1] + offset_y, pos[2] + offset_z,
                name, fontsize=10, fontweight=fontweight)

    ax.set_xlabel('X (Anterior)', fontsize=11)
    ax.set_ylabel('Y (Left)', fontsize=11)
    ax.set_zlabel('Z (Superior)', fontsize=11)
    all_pos = np.array([ELECTRODE_POSITIONS[n] for n in ALL_ELECTRODES])
    margin = 0.3
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax.set_zlim(all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin)

    ax.view_init(elev=25, azim=45)

    legend_elements = [
        mpatches.Patch(color='#27ae60', label='WCT (Reference)'),
        mpatches.Patch(color='#3498db', label='Precordial (V1-V6)'),
        mpatches.Patch(color='#9b59b6', label='Virtual Midpoints'),
        Line2D([0], [0], color='#bdc3c7', lw=1.5, alpha=0.5, label='Supergraph Edges (156 total)'),
        Line2D([0], [0], color='#e74c3c', lw=4, label='ICM Lead (V2 → V3)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    ax.text2D(0.02, 0.02, 'ICM (Implantable Cardiac Monitor) lead:\n'
              'Measures voltage difference V(V3) - V(V2)\n'
              'Selected from the 156-edge supergraph',
              transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_training_overview_figure():
    """
    Create figure showing the multi-task training pipeline.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.set_title('A. Lead Dropout Training Strategy', fontweight='bold', fontsize=12)

    configs = [
        ('Full 12-lead', list(range(12)), 1.0),
        ('10-lead (dropout)', [0,1,2,3,4,5,6,7,9,11], 0.8),
        ('8-lead (dropout)', [0,1,2,3,4,5,7,10], 0.6),
        ('6-lead (limb only)', [0,1,2,3,4,5], 0.4),
    ]

    leads = LEAD_NAMES_12
    y_positions = np.arange(len(leads))

    for i, (name, indices, x_pos) in enumerate(configs):
        for j, lead in enumerate(leads):
            if j in indices:
                color = LEAD_COLORS[get_lead_type(lead)]
                ax1.scatter(x_pos, j, c=color, s=100, marker='s', edgecolors='black', linewidths=0.5)
            else:
                ax1.scatter(x_pos, j, c='#f0f0f0', s=80, marker='s', edgecolors='gray',
                           linewidths=0.5, alpha=0.5)

        ax1.text(x_pos, -0.8, name, ha='center', fontsize=9, rotation=0)
        ax1.text(x_pos, 12.3, f'{len(indices)} leads', ha='center', fontsize=8, color='gray')

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(leads)
    ax1.set_xlim(0.2, 1.2)
    ax1.set_ylim(-1.5, 13)
    ax1.set_xlabel('Training Examples (varying input configurations)')
    ax1.axhline(5.5, color='gray', linestyle='--', alpha=0.3)
    ax1.text(1.25, 2.5, 'Limb', fontsize=9, color='gray', rotation=90, va='center')
    ax1.text(1.25, 8.5, 'Precordial', fontsize=9, color='gray', rotation=90, va='center')
    ax1.set_xticks([])

    ax2 = axes[1]
    ax2.set_title('B. Multi-Task Learning Objective', fontweight='bold', fontsize=12)
    ax2.axis('off')
    loss_text = """L_total = w_cls * L_cls + w_recon * L_recon

L_cls = BCE(y_pred, y_true)
      = -[y * log(y_pred) + (1-y) * log(1-y_pred)]

L_recon = (1/|Q|) * sum ||s_pred - s_true||^2

where:
  y_pred = predicted SHD probability
  y_true = ground truth label (0 or 1)
  Q = set of query coordinates
  s_pred, s_true = predicted/true signals

Defaults: w_cls = 1.0, w_recon = 0.1"""

    ax2.text(0.5, 0.5, loss_text, transform=ax2.transAxes, fontsize=10,
            va='center', ha='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f8f8', edgecolor='gray', alpha=0.9))

    plt.tight_layout()
    return fig


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("Creating ECG Graph V2 figures...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    print("\n1. Creating full/subgraph comparison figure...")
    fig1 = create_full_connected_figure()
    fig1.savefig(os.path.join(output_dir, 'ecg_graph_panel_comparison.png'), dpi=300, bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'ecg_graph_panel_comparison.pdf'), bbox_inches='tight')
    print("   Saved: ecg_graph_panel_comparison.png/pdf")

    print("\n2. Creating bidirectional physics figure...")
    fig2 = create_bidirectional_physics_figure()
    fig2.savefig(os.path.join(output_dir, 'ecg_bidirectional_physics.png'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'ecg_bidirectional_physics.pdf'), bbox_inches='tight')
    print("   Saved: ecg_bidirectional_physics.png/pdf")

    print("\n3. Creating 3D electrode graph figure (12-lead)...")
    fig3 = create_3d_electrode_graph_figure()
    fig3.savefig(os.path.join(output_dir, 'ecg_3d_electrode_graph_12lead.png'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'ecg_3d_electrode_graph_12lead.pdf'), bbox_inches='tight')
    print("   Saved: ecg_3d_electrode_graph_12lead.png/pdf")

    print("\n4. Creating fully connected supergraph figure...")
    fig4 = create_fully_connected_3d_figure()
    fig4.savefig(os.path.join(output_dir, 'ecg_3d_supergraph_fully_connected.png'), dpi=300, bbox_inches='tight')
    fig4.savefig(os.path.join(output_dir, 'ecg_3d_supergraph_fully_connected.pdf'), bbox_inches='tight')
    print("   Saved: ecg_3d_supergraph_fully_connected.png/pdf")

    print("\n5. Creating training overview figure...")
    fig5 = create_training_overview_figure()
    fig5.savefig(os.path.join(output_dir, 'ecg_training_overview.png'), dpi=300, bbox_inches='tight')
    fig5.savefig(os.path.join(output_dir, 'ecg_training_overview.pdf'), bbox_inches='tight')
    print("   Saved: ecg_training_overview.png/pdf")

    plt.close('all')

    print("\n" + "=" * 60)
    print("All figures created successfully!")
    print("\nGenerated files in figures/:")
    print("  - ecg_graph_panel_comparison.pdf       (Supergraph vs 12-lead vs 6-lead panels)")
    print("  - ecg_bidirectional_physics.pdf        (Bidirectional edge physics)")
    print("  - ecg_3d_electrode_graph_12lead.pdf    (3D electrode graph with 12 leads)")
    print("  - ecg_3d_supergraph_fully_connected.pdf (3D fully connected supergraph)")
    print("  - ecg_training_overview.pdf            (Multi-task training)")


if __name__ == '__main__':
    main()

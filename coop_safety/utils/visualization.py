"""Visualization module for safety constraint analysis.

Generates publication-quality figures showing:
1. RiskMap heatmap with agent positions
2. RiskGraph as a network diagram
3. Feasible region shrinkage process
4. Full scene overview combining all layers
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection
from shapely.geometry import Polygon as ShapelyPolygon
from pathlib import Path
from typing import Optional

import sys
_proj_root = str(Path(__file__).parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from coop_safety.interface import (
    PerceptionResult, SafeActionSpace, ThreeLayerRisk,
    RiskLevel, ConstraintMode, VehicleState, Agent,
    SafetyConstraintModule,
)


# Color scheme
COLORS = {
    RiskLevel.HIGH: '#d32f2f',
    RiskLevel.MEDIUM: '#ff9800',
    RiskLevel.LOW: '#4caf50',
    'ego': '#1565c0',
    'vehicle': '#37474f',
    'pedestrian': '#e91e63',
    'cyclist': '#9c27b0',
    'phantom': '#78909c',
    'feasible': '#2196f3',
    'feasible_alpha': 0.25,
    'excluded': '#f44336',
}


def plot_scene_overview(perception: PerceptionResult,
                        risk: ThreeLayerRisk,
                        safe_space: SafeActionSpace,
                        title: str = "",
                        save_path: Optional[str] = None,
                        figsize: tuple = (16, 12)):
    """Generate a 2×2 figure with all four views."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title or f"Safety Constraint Analysis (mode={safe_space.mode.value})",
                 fontsize=14, fontweight='bold')

    plot_risk_map(perception, risk, ax=axes[0, 0])
    plot_risk_graph(perception, risk, ax=axes[0, 1])
    plot_feasible_region(perception, safe_space, ax=axes[1, 0])
    plot_constraint_summary(perception, risk, safe_space, ax=axes[1, 1])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)
    return fig


def plot_risk_map(perception: PerceptionResult, risk: ThreeLayerRisk,
                  ax=None, title="RiskMap: Spatial Risk"):
    """Plot RiskMap as colored grid cells."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ego = perception.ego

    # Draw risk cells
    for region in risk.risk_map:
        color = COLORS[region.risk_level]
        alpha = 0.15 + 0.35 * region.risk_score
        poly = plt.Polygon(region.polygon, facecolor=color, edgecolor='none', alpha=alpha)
        ax.add_patch(poly)

    # Draw agents
    _draw_agents(ax, perception)

    # Legend
    patches = [mpatches.Patch(color=COLORS[RiskLevel.HIGH], alpha=0.5, label='HIGH risk'),
               mpatches.Patch(color=COLORS[RiskLevel.MEDIUM], alpha=0.4, label='MEDIUM risk'),
               mpatches.Patch(color=COLORS[RiskLevel.LOW], alpha=0.2, label='LOW risk')]
    ax.legend(handles=patches, loc='upper right', fontsize=8)

    _set_axis(ax, ego, title)


def plot_risk_graph(perception: PerceptionResult, risk: ThreeLayerRisk,
                    ax=None, title="RiskGraph: Pairwise Conflicts"):
    """Plot RiskGraph as edges between agents."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ego = perception.ego
    _draw_agents(ax, perception)

    # Build position lookup
    positions = {"ego": np.array([ego.x, ego.y])}
    for agent in perception.agents:
        positions[agent.state.id] = np.array([agent.state.x, agent.state.y])

    # Draw conflict edges
    for edge in risk.risk_graph:
        p1 = positions.get(edge.agent_a_id)
        p2 = positions.get(edge.agent_b_id)
        if p1 is None or p2 is None:
            continue

        # Color by collision probability, width by 1/TTC
        prob = edge.collision_probability
        ttc = min(edge.ttc, 10)
        width = max(0.5, 3.0 * (1.0 - ttc / 10.0))
        color = plt.cm.Reds(0.3 + 0.7 * prob)

        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linewidth=width, alpha=0.7, zorder=1)

        # Label TTC at midpoint
        mid = (p1 + p2) / 2
        if ttc < 5:
            ax.text(mid[0], mid[1], f"TTC={ttc:.1f}s",
                    fontsize=6, ha='center', color='red',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))

    n_edges = len(risk.risk_graph)
    ax.set_title(f"{title} ({n_edges} edges)", fontsize=10)
    _set_axis(ax, ego, "")
    ax.set_title(f"{title} ({n_edges} conflicts)", fontsize=10)


def plot_feasible_region(perception: PerceptionResult, safe_space: SafeActionSpace,
                         ax=None, title="Feasible Region"):
    """Plot the constrained feasible region."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ego = perception.ego

    # Draw feasible region
    if len(safe_space.feasible_region) >= 3:
        poly = plt.Polygon(safe_space.feasible_region,
                           facecolor=COLORS['feasible'],
                           edgecolor='#0d47a1',
                           alpha=COLORS['feasible_alpha'],
                           linewidth=2, zorder=2)
        ax.add_patch(poly)

    _draw_agents(ax, perception)

    # Mode indicator
    mode_colors = {
        ConstraintMode.NORMAL: '#4caf50',
        ConstraintMode.CONSERVATIVE: '#ff9800',
        ConstraintMode.MINIMUM_HARM: '#d32f2f',
    }
    mode = safe_space.mode
    mode_color = mode_colors.get(mode, '#999')
    area = ShapelyPolygon(safe_space.feasible_region).area if len(safe_space.feasible_region) >= 3 else 0

    info = f"Mode: {mode.value}\nArea: {area:.0f} m²\nTTC: {safe_space.safety_margin_ttc:.1f}s"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=mode_color, alpha=0.3))

    _set_axis(ax, ego, title)


def plot_constraint_summary(perception: PerceptionResult, risk: ThreeLayerRisk,
                            safe_space: SafeActionSpace, ax=None,
                            title="Constraint Summary"):
    """Text summary of the constraint reasoning chain."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.axis('off')
    lines = []
    lines.append(f"Scene: {len(perception.agents)} agents")
    lines.append(f"Risk: {sum(1 for r in risk.risk_map if r.risk_level == RiskLevel.HIGH)} HIGH, "
                 f"{sum(1 for r in risk.risk_map if r.risk_level == RiskLevel.MEDIUM)} MEDIUM")
    lines.append(f"Conflicts: {len(risk.risk_graph)} edges, {len(risk.risk_events)} events")
    lines.append(f"")
    lines.append(f"Mode: {safe_space.mode.value}")
    area = ShapelyPolygon(safe_space.feasible_region).area if len(safe_space.feasible_region) >= 3 else 0
    lines.append(f"Feasible area: {area:.0f} m²")
    lines.append(f"Min TTC: {safe_space.safety_margin_ttc:.2f}s")
    lines.append(f"Future feasible: {'Yes' if safe_space.future_feasible else 'No'}")
    lines.append(f"")
    lines.append("Reasoning:")
    for r in safe_space.reasoning[:6]:
        lines.append(f"  • {r[:80]}")

    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.8))
    ax.set_title(title, fontsize=10)


def _draw_agents(ax, perception: PerceptionResult):
    """Draw ego and all agents as oriented rectangles."""
    ego = perception.ego

    # Ego vehicle (blue)
    _draw_vehicle(ax, ego, COLORS['ego'], label='Ego', zorder=10)

    # Other agents
    for agent in perception.agents:
        s = agent.state
        if s.vehicle_type == 'pedestrian':
            ax.plot(s.x, s.y, 'o', color=COLORS['pedestrian'], markersize=5, zorder=8)
        elif s.vehicle_type in ('bicycle', 'motorcycle'):
            ax.plot(s.x, s.y, '^', color=COLORS['cyclist'], markersize=5, zorder=8)
        elif not agent.is_visible:
            _draw_vehicle(ax, s, COLORS['phantom'], zorder=6, alpha=0.4)
        else:
            _draw_vehicle(ax, s, COLORS['vehicle'], zorder=7)

        # Velocity arrow
        if s.velocity > 0.5:
            ax.arrow(s.x, s.y, s.vx * 0.5, s.vy * 0.5,
                     head_width=0.5, head_length=0.3,
                     fc='gray', ec='gray', alpha=0.5, zorder=5)


def _draw_vehicle(ax, state: VehicleState, color: str,
                  label: str = "", zorder: int = 7, alpha: float = 0.8):
    """Draw an oriented rectangle for a vehicle."""
    L, W = state.length, state.width
    cos_h, sin_h = np.cos(state.heading), np.sin(state.heading)

    corners = np.array([[-L/2, -W/2], [L/2, -W/2], [L/2, W/2], [-L/2, W/2]])
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = corners @ rotation.T + np.array([state.x, state.y])

    poly = plt.Polygon(rotated, facecolor=color, edgecolor='black',
                        alpha=alpha, linewidth=0.5, zorder=zorder)
    ax.add_patch(poly)

    # Direction indicator
    front = np.array([state.x + L/2 * cos_h, state.y + L/2 * sin_h])
    ax.plot(*front, 'o', color='white', markersize=2, zorder=zorder + 1)

    if label:
        ax.text(state.x, state.y + W, label, fontsize=7, ha='center',
                color=color, fontweight='bold', zorder=zorder + 1)


def _set_axis(ax, ego: VehicleState, title: str, radius: float = 60):
    """Set axis limits centered on ego."""
    ax.set_xlim(ego.x - radius, ego.x + radius)
    ax.set_ylim(ego.y - radius, ego.y + radius)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)


def generate_paper_figures(output_dir: str = "/raid/xuyifan/jiqiuyu/paper/figures"):
    """Generate all figures for the paper using DAIR-V2X data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from experiments.dairv2x_loader import DAIRv2xLoader

    # Load a few representative frames
    data_dir = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
    loader = DAIRv2xLoader(data_dir, ego_speed=10.0)

    module = SafetyConstraintModule()

    # Pick frames with different characteristics
    frame_indices = [0, 100, 500, 1000, 2000, 3000, 4000, 5000]
    for idx in frame_indices:
        if idx >= len(loader):
            continue

        perception = loader.load_frame(idx)
        n_agents = len(perception.agents)

        # Run our method
        safe_space = module.constrain(perception)
        risk = module.assess_risk(perception)

        frame_id = loader.frames[idx]
        save_path = str(out / f"scene_{frame_id}_{n_agents}agents.png")

        plot_scene_overview(
            perception, risk, safe_space,
            title=f"Frame {frame_id} — {n_agents} agents — Mode: {safe_space.mode.value}",
            save_path=save_path,
        )

    print(f"\nGenerated {len(frame_indices)} figures in {out}")


if __name__ == "__main__":
    generate_paper_figures()

"""Cooperative vs Non-Cooperative perception comparison.

Shows the advantage of V2X cooperative perception for safety constraints.
- Cooperative: all agents visible (V2I fusion)
- Non-cooperative: only ego-side detections (remove infrastructure-only detections)

Uses DAIR-V2X which has both vehicle-side and infrastructure-side labels.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import SafetyConstraintModule, PerceptionResult, Agent
from experiments.dairv2x_loader import DAIRv2xLoader
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario, ScenarioMetrics


def make_non_cooperative(perception: PerceptionResult, drop_rate: float = 0.4) -> PerceptionResult:
    """Simulate non-cooperative perception by removing agents that would
    only be visible to the infrastructure sensor.

    In real V2I cooperation, infrastructure detects vehicles that ego can't see
    (occluded, behind buildings, etc.). Without cooperation, these agents are
    invisible to ego.

    Args:
        perception: Full cooperative perception
        drop_rate: Fraction of agents to remove (simulating no infrastructure data)

    Returns:
        Degraded perception with fewer agents
    """
    rng = np.random.RandomState(42)

    # Keep agents that are "close and likely visible to ego"
    ego = perception.ego
    kept = []
    dropped = []

    for agent in perception.agents:
        dist = np.sqrt((agent.state.x - ego.x)**2 + (agent.state.y - ego.y)**2)

        # Agents close to ego are likely visible regardless
        if dist < 15:
            kept.append(agent)
            continue

        # Far agents or low-confidence agents are infrastructure contributions
        if agent.confidence < 0.7 or agent.source == "infrastructure_only":
            dropped.append(agent)
            continue

        # For remaining agents, randomly drop some to simulate limited ego FoV
        if rng.random() < drop_rate * (dist / 80.0):  # Higher drop rate for far agents
            dropped.append(agent)
        else:
            kept.append(agent)

    return PerceptionResult(
        timestamp=perception.timestamp,
        ego=perception.ego,
        agents=kept,
        lanes=perception.lanes,
        blind_spots=perception.blind_spots,
    ), len(dropped)


def main():
    output_dir = Path(__file__).parent / "results" / f"coop_vs_noncoop_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COOPERATIVE vs NON-COOPERATIVE PERCEPTION COMPARISON")
    print("=" * 70)

    data_dir = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
    loader = DAIRv2xLoader(data_dir, ego_speed=10.0)

    module = SafetyConstraintModule()

    # Sample 200 frames evenly
    n_sample = 200
    indices = np.linspace(0, len(loader) - 1, n_sample, dtype=int)

    results_coop = []
    results_noncoop = []
    agent_counts = []

    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f"\n[{i}/{n_sample}]")

        perception_coop = loader.load_frame(idx)
        perception_noncoop, n_dropped = make_non_cooperative(perception_coop)

        config = ScenarioConfig(
            name=f"frame_{loader.frames[idx]}",
            description=f"{len(perception_coop.agents)} agents",
            seed=42 + i, difficulty="hard",
        )

        # Evaluate with cooperative perception
        m_coop = evaluate_method_on_scenario(module, perception_coop, config)
        results_coop.append(m_coop)

        # Evaluate with non-cooperative (degraded) perception
        config_nc = ScenarioConfig(
            name=f"frame_{loader.frames[idx]}_noncoop",
            description=f"{len(perception_noncoop.agents)} agents (no infra)",
            seed=42 + i, difficulty="hard",
        )
        m_noncoop = evaluate_method_on_scenario(module, perception_noncoop, config_nc)
        results_noncoop.append(m_noncoop)

        agent_counts.append({
            "frame": loader.frames[idx],
            "coop_agents": len(perception_coop.agents),
            "noncoop_agents": len(perception_noncoop.agents),
            "dropped": n_dropped,
        })

    # Summary
    print("\n" + "=" * 70)
    print("COOPERATIVE vs NON-COOPERATIVE RESULTS")
    print("=" * 70)

    def summarize(metrics, label):
        areas = [m.feasible_area for m in metrics]
        tights = [m.constraint_tightening_ratio for m in metrics]
        ttcs = [min(m.min_ttc, 100) for m in metrics]
        mh = sum(1 for m in metrics if m.mode == "minimum_harm")
        print(f"\n{label}:")
        print(f"  Avg Feasible Area: {np.mean(areas):.1f} m² (±{np.std(areas):.1f})")
        print(f"  Avg Constraint %:  {np.mean(tights):.1%}")
        print(f"  Avg Min TTC:       {np.mean(ttcs):.2f}s")
        print(f"  Min-Harm triggers: {mh}/{len(metrics)} ({100*mh/len(metrics):.1f}%)")
        return {
            "avg_area": float(np.mean(areas)),
            "std_area": float(np.std(areas)),
            "avg_tightening": float(np.mean(tights)),
            "avg_ttc": float(np.mean(ttcs)),
            "min_harm_count": mh,
            "min_harm_rate": mh / len(metrics),
        }

    stats_coop = summarize(results_coop, "Cooperative (V2I)")
    stats_noncoop = summarize(results_noncoop, "Non-Cooperative (ego only)")

    avg_dropped = np.mean([c["dropped"] for c in agent_counts])
    print(f"\nAvg agents dropped (no V2I): {avg_dropped:.1f}")

    # Safety gap analysis
    area_diff = stats_noncoop["avg_area"] - stats_coop["avg_area"]
    mh_diff = stats_noncoop["min_harm_count"] - stats_coop["min_harm_count"]
    print(f"\nCooperative advantage:")
    print(f"  Area difference: {area_diff:+.1f} m² (noncoop has {'more' if area_diff > 0 else 'less'} area → {'less' if area_diff > 0 else 'more'} safe)")
    print(f"  Min-harm difference: {mh_diff:+d} (noncoop triggers {'more' if mh_diff > 0 else 'fewer'} min-harm)")

    # Save
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "DAIR-V2X-C",
            "n_frames": n_sample,
            "description": "Compare safety constraints with/without V2I cooperative perception",
        },
        "cooperative": stats_coop,
        "non_cooperative": stats_noncoop,
        "agent_counts": agent_counts,
        "metrics_coop": [asdict(m) for m in results_coop],
        "metrics_noncoop": [asdict(m) for m in results_noncoop],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    summary_lines = [
        "# Cooperative vs Non-Cooperative Perception\n",
        f"DAIR-V2X-C, {n_sample} frames\n",
        "",
        "| Setting | Avg Area | Constraint % | Min TTC | Min-Harm Rate |",
        "|---------|---------|-------------|---------|---------------|",
        f"| Cooperative (V2I) | {stats_coop['avg_area']:.1f} m² | {stats_coop['avg_tightening']:.1%} | {stats_coop['avg_ttc']:.2f}s | {stats_coop['min_harm_rate']:.1%} |",
        f"| Non-Cooperative | {stats_noncoop['avg_area']:.1f} m² | {stats_noncoop['avg_tightening']:.1%} | {stats_noncoop['avg_ttc']:.2f}s | {stats_noncoop['min_harm_rate']:.1%} |",
        f"\nAvg agents visible: Coop={np.mean([c['coop_agents'] for c in agent_counts]):.0f}, NonCoop={np.mean([c['noncoop_agents'] for c in agent_counts]):.0f}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

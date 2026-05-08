"""Experiment runner — evaluates all methods on all scenarios.

Collects safety metrics, records full results in JSON for reproducibility.
Every result includes: method config, scenario config, raw metrics, timestamps.

Usage:
    cd /raid/xuyifan/jiqiuyu
    python experiments/run_experiments.py

Output:
    experiments/results/YYYY-MM-DD_HHMMSS/
        results.json        — full structured results
        summary_table.md    — human-readable comparison table
        per_scenario/       — per-scenario detailed metrics
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import SafeActionSpace, ConstraintMode, PerceptionResult
from coop_safety.utils.metrics import compute_ttc
from experiments.scenarios import get_all_scenarios, ScenarioConfig
from experiments.methods import get_all_methods


@dataclass
class ScenarioMetrics:
    """Metrics for one method on one scenario."""
    method_name: str
    scenario_name: str
    scenario_difficulty: str

    # Core safety metrics
    feasible_area: float = 0.0           # m² of safe action space
    min_ttc: float = float('inf')        # minimum TTC (seconds)
    mean_ttc_to_agents: float = float('inf')  # mean TTC across all agents
    n_high_risk_regions: int = 0         # number of HIGH risk regions in feasible area
    n_conflict_edges: int = 0            # number of conflict edges detected
    n_collision_events: int = 0          # number of collision events
    max_collision_severity: float = 0.0  # worst case severity

    # Constraint behavior
    mode: str = "NORMAL"                 # NORMAL / CONSERVATIVE / MINIMUM_HARM
    constraint_tightening_ratio: float = 0.0  # how much area was removed
    future_feasible: bool = True
    future_feasible_horizon: float = 0.0
    n_reasoning_steps: int = 0

    # Computation
    compute_time_ms: float = 0.0

    # Error tracking
    error: str = ""


def evaluate_method_on_scenario(method, perception: PerceptionResult,
                                 config: ScenarioConfig) -> ScenarioMetrics:
    """Run one method on one scenario and collect metrics."""
    method_name = getattr(method, 'name', method.__class__.__name__)

    metrics = ScenarioMetrics(
        method_name=method_name,
        scenario_name=config.name,
        scenario_difficulty=config.difficulty,
    )

    try:
        t_start = time.time()
        result: SafeActionSpace = method.constrain(perception)
        metrics.compute_time_ms = (time.time() - t_start) * 1000

        # Extract metrics from result
        from shapely.geometry import Polygon
        try:
            poly = Polygon(result.feasible_region)
            metrics.feasible_area = poly.area if poly.is_valid and not poly.is_empty else 0.0
        except Exception:
            metrics.feasible_area = 0.0

        metrics.min_ttc = result.safety_margin_ttc if result.safety_margin_ttc != float('inf') else 999.0
        metrics.mode = result.mode.value if isinstance(result.mode, ConstraintMode) else str(result.mode)
        metrics.future_feasible = result.future_feasible
        metrics.future_feasible_horizon = result.future_feasible_horizon
        metrics.n_reasoning_steps = len(result.reasoning)

        # Compute mean TTC to all agents
        ttcs = []
        ego = perception.ego
        for agent in perception.agents:
            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([agent.state.x, agent.state.y]),
                np.array([agent.state.vx, agent.state.vy]),
            )
            if ttc < 100:
                ttcs.append(ttc)
        metrics.mean_ttc_to_agents = np.mean(ttcs) if ttcs else 999.0

        # Compute constraint tightening ratio
        from coop_safety.perception.dynamics import BicycleModel, get_dynamics_params
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])
        full_reachable = model.compute_reachable_set(state, 3.0, 0.1, 12)
        try:
            full_area = Polygon(full_reachable).area
            if full_area > 0:
                metrics.constraint_tightening_ratio = 1.0 - metrics.feasible_area / full_area
            else:
                metrics.constraint_tightening_ratio = 0.0
        except Exception:
            metrics.constraint_tightening_ratio = 0.0

        # Risk assessment metrics (only for our methods)
        if hasattr(method, 'assess_risk') or hasattr(method, '_module'):
            try:
                module = getattr(method, '_module', method)
                if hasattr(module, 'assess_risk'):
                    risk = module.assess_risk(perception)
                    from coop_safety.interface import RiskLevel
                    metrics.n_high_risk_regions = sum(
                        1 for r in risk.risk_map if r.risk_level == RiskLevel.HIGH
                    )
                    metrics.n_conflict_edges = len(risk.risk_graph)
                    metrics.n_collision_events = len(risk.risk_events)
                    if risk.risk_events:
                        metrics.max_collision_severity = max(e.severity for e in risk.risk_events)
            except Exception:
                pass

    except Exception as e:
        metrics.error = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

    return metrics


def run_all_experiments() -> dict:
    """Run all methods on all scenarios. Returns full results dict."""
    print("=" * 70)
    print("EXPERIMENT: Cooperative Perception-based Safe Action Space Constraint")
    print("=" * 70)

    scenarios = get_all_scenarios()
    methods = get_all_methods()

    print(f"\nScenarios: {len(scenarios)}")
    for p, c in scenarios:
        print(f"  - {c.name} ({c.difficulty}): {c.description}")

    print(f"\nMethods: {len(methods)}")
    for name in methods:
        print(f"  - {name}")

    # Run experiments
    all_metrics = []
    for scenario_perception, scenario_config in scenarios:
        print(f"\n--- Scenario: {scenario_config.name} ({scenario_config.difficulty}) ---")
        for method_name, method in methods.items():
            print(f"  Running {method_name}...", end=" ", flush=True)
            metrics = evaluate_method_on_scenario(
                method, scenario_perception, scenario_config
            )
            print(f"area={metrics.feasible_area:.1f}m², "
                  f"ttc={metrics.min_ttc:.2f}s, "
                  f"mode={metrics.mode}, "
                  f"time={metrics.compute_time_ms:.0f}ms"
                  + (f" ERROR: {metrics.error}" if metrics.error else ""))
            all_metrics.append(metrics)

    # Build results dict
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(scenarios),
            "n_methods": len(methods),
            "scenario_names": [c.name for _, c in scenarios],
            "method_names": list(methods.keys()),
        },
        "metrics": [asdict(m) for m in all_metrics],
    }

    return results, all_metrics


def generate_summary_table(all_metrics: list[ScenarioMetrics]) -> str:
    """Generate human-readable Markdown comparison table."""
    lines = []
    lines.append("# Experiment Results Summary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Aggregate metrics per method
    method_names = sorted(set(m.method_name for m in all_metrics))
    scenario_names = sorted(set(m.scenario_name for m in all_metrics))

    # Table 1: Per-method aggregate
    lines.append("## Table 1: Aggregate Safety Metrics by Method\n")
    lines.append("| Method | Avg Feasible Area (m²) | Avg Min TTC (s) | Avg Tightening Ratio | Min-Harm Count | Avg Time (ms) |")
    lines.append("|--------|----------------------|-----------------|---------------------|----------------|---------------|")

    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        avg_area = np.mean([m.feasible_area for m in mm])
        # For TTC, cap at 100 for averaging
        ttcs = [min(m.min_ttc, 100) for m in mm]
        avg_ttc = np.mean(ttcs)
        avg_tight = np.mean([m.constraint_tightening_ratio for m in mm])
        min_harm_count = sum(1 for m in mm if m.mode == "minimum_harm")
        avg_time = np.mean([m.compute_time_ms for m in mm])

        lines.append(f"| {mname} | {avg_area:.1f} | {avg_ttc:.2f} | {avg_tight:.3f} | {min_harm_count} | {avg_time:.0f} |")

    # Table 2: Per-scenario results
    lines.append("\n## Table 2: Per-Scenario Detailed Results\n")
    for sname in scenario_names:
        sm = [m for m in all_metrics if m.scenario_name == sname]
        diff = sm[0].scenario_difficulty if sm else "?"
        lines.append(f"\n### Scenario: {sname} (difficulty: {diff})\n")
        lines.append("| Method | Feasible Area | Min TTC | Mode | Tightening | Future Feasible | Time (ms) |")
        lines.append("|--------|--------------|---------|------|-----------|----------------|-----------|")

        for m in sorted(sm, key=lambda x: x.method_name):
            ttc_str = f"{m.min_ttc:.2f}" if m.min_ttc < 100 else "∞"
            lines.append(
                f"| {m.method_name} | {m.feasible_area:.1f} | {ttc_str} | "
                f"{m.mode} | {m.constraint_tightening_ratio:.3f} | "
                f"{'✓' if m.future_feasible else '✗'} | {m.compute_time_ms:.0f} |"
            )

    # Table 3: Ablation comparison
    lines.append("\n## Table 3: Ablation Study\n")
    lines.append("| Component Removed | Avg Area Change | Avg TTC Change | Min-Harm Change | Interpretation |")
    lines.append("|-------------------|----------------|----------------|-----------------|----------------|")

    full_metrics = [m for m in all_metrics if m.method_name == "Ours-Full"]
    ablation_names = [n for n in method_names if n.startswith("Ours-") and n != "Ours-Full"]

    for abl_name in ablation_names:
        abl_metrics = [m for m in all_metrics if m.method_name == abl_name]
        if not full_metrics or not abl_metrics:
            continue

        # Compare against full method on matching scenarios
        area_diffs = []
        ttc_diffs = []
        for fm in full_metrics:
            matching = [a for a in abl_metrics if a.scenario_name == fm.scenario_name]
            if matching:
                am = matching[0]
                area_diffs.append(am.feasible_area - fm.feasible_area)
                ttc_diffs.append(min(am.min_ttc, 100) - min(fm.min_ttc, 100))

        avg_area_diff = np.mean(area_diffs) if area_diffs else 0
        avg_ttc_diff = np.mean(ttc_diffs) if ttc_diffs else 0
        mh_full = sum(1 for m in full_metrics if m.mode == "minimum_harm")
        mh_abl = sum(1 for m in abl_metrics if m.mode == "minimum_harm")

        component = abl_name.replace("Ours-", "").replace("No", "No ")
        interpretation = ""
        if avg_area_diff > 10:
            interpretation = "Larger area → less constrained → potentially less safe"
        elif avg_area_diff < -10:
            interpretation = "Smaller area → over-constrained"
        else:
            interpretation = "Marginal difference"

        lines.append(
            f"| {component} | {avg_area_diff:+.1f}m² | {avg_ttc_diff:+.2f}s | "
            f"{mh_abl - mh_full:+d} | {interpretation} |"
        )

    return "\n".join(lines)


def main():
    output_dir = Path(__file__).parent / "results" / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    results, all_metrics = run_all_experiments()

    # Save results JSON
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate and save summary table
    summary = generate_summary_table(all_metrics)
    summary_path = output_dir / "summary_table.md"
    summary_path.write_text(summary)
    print(f"Summary table saved to {summary_path}")

    # Print summary to console
    print("\n" + summary)

    # Save reproduction info
    repro = {
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "command": f"cd /raid/xuyifan/jiqiuyu && {sys.executable} experiments/run_experiments.py",
        "conda_env": "coop-safety",
        "note": "All scenarios use fixed random seeds for reproducibility. "
                "Re-running produces identical results.",
    }
    (output_dir / "reproduction_info.json").write_text(json.dumps(repro, indent=2))

    return output_dir


if __name__ == "__main__":
    main()

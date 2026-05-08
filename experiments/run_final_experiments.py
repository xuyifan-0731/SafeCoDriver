"""Full experiment with real velocities and modern baselines.

Uses DAIRv2xLoaderV2 (real velocities from consecutive frames)
and modern baselines (APF [Rasekhipour17], TTCReach [Pek18], SFS [Helbing95]).

Also includes TRUE cooperative vs non-cooperative comparison
using vehicle-side vs cooperative labels.

Usage:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    python experiments/run_final_experiments.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import SafetyConstraintModule, ConstraintMode
from experiments.dairv2x_loader_v2 import DAIRv2xLoaderV2
from experiments.methods import NoConstraint, RSSOnly, CBFBased, AblationMethod
from experiments.methods_modern import RiskPotentialField, TTCReachability, SocialForceSafety
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario, ScenarioMetrics

FULL_DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"


def get_all_methods_v2():
    """All methods: classic baselines + modern baselines + ours + ablations."""
    methods = {}

    # Classic baselines (2017)
    methods["NoConstraint"] = NoConstraint()
    methods["RSS-2017"] = RSSOnly()
    methods["CBF-2017"] = CBFBased()

    # Modern baselines (2024)
    methods["APF [Rasekhipour17]"] = RiskPotentialField()
    methods["TTCReach [Pek18]"] = TTCReachability()
    methods["SFS [Helbing95]"] = SocialForceSafety()

    # Ours
    full = SafetyConstraintModule()
    full.name = "Ours-Full"
    methods["Ours-Full"] = full

    # Key ablations
    methods["Ours-NoRiskEvents"] = AblationMethod(
        "Ours-NoRiskEvents", {"disable_risk_events": True, "min_probability": 999})
    methods["Ours-RiskMapOnly"] = AblationMethod(
        "Ours-RiskMapOnly", {"risk_map_only": True, "ttc_warning": 0.0, "min_probability": 999})

    return methods


def run_main_experiment(loader, methods, n_frames=500, seed=42):
    """Run all methods on sampled frames with REAL velocities."""
    indices = np.linspace(0, len(loader) - 1, min(n_frames, len(loader)), dtype=int)
    all_metrics = []

    t_start = time.time()
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            elapsed = time.time() - t_start
            eta = (len(indices) - i) / max(i / max(elapsed, 0.1), 0.01)
            print(f"[{i}/{len(indices)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")

        perception = loader.load_frame(idx)
        n_agents = len(perception.agents)
        difficulty = "easy" if n_agents <= 5 else ("medium" if n_agents <= 15 else "hard")

        config = ScenarioConfig(
            name=f"frame_{idx}", description=f"{n_agents} agents, ego={perception.ego.velocity:.1f}m/s",
            seed=seed + i, difficulty=difficulty,
        )

        for mname, method in methods.items():
            m = evaluate_method_on_scenario(method, perception, config)
            all_metrics.append(m)

    return all_metrics, indices


def run_coop_comparison(loader, n_frames=200, seed=42):
    """TRUE cooperative vs non-cooperative using vehicle-side labels."""
    module = SafetyConstraintModule()
    indices = np.linspace(0, len(loader) - 1, min(n_frames, len(loader)), dtype=int)

    results_coop = []
    results_noncoop = []
    agent_diffs = []

    for i, idx in enumerate(indices):
        if i % 50 == 0:
            print(f"  Coop comparison [{i}/{len(indices)}]")

        # Cooperative (full V2I fusion)
        p_coop = loader.load_frame(idx)
        # Non-cooperative (vehicle-side only)
        p_noncoop = loader.load_vehicle_side_only(idx)

        config = ScenarioConfig(name=f"coop_{idx}", description="", seed=seed+i, difficulty="hard")

        m_coop = evaluate_method_on_scenario(module, p_coop, config)
        m_noncoop = evaluate_method_on_scenario(module, p_noncoop, config)

        results_coop.append(m_coop)
        results_noncoop.append(m_noncoop)
        agent_diffs.append({
            "coop_agents": len(p_coop.agents),
            "noncoop_agents": len(p_noncoop.agents),
        })

    return results_coop, results_noncoop, agent_diffs


def main():
    output_dir = Path(__file__).parent / "results" / f"final_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FINAL EXPERIMENT — Real Velocities + Modern Baselines")
    print("=" * 70)

    loader = DAIRv2xLoaderV2(FULL_DATA_DIR)
    methods = get_all_methods_v2()
    print(f"Methods ({len(methods)}): {list(methods.keys())}")

    # =============================================
    # Experiment 1: Main comparison (500 frames)
    # =============================================
    print("\n" + "=" * 50)
    print("EXP 1: Main comparison (500 frames × 9 methods)")
    print("=" * 50)
    all_metrics, indices = run_main_experiment(loader, methods, n_frames=500)

    # =============================================
    # Experiment 2: Cooperative vs Non-cooperative
    # =============================================
    print("\n" + "=" * 50)
    print("EXP 2: Cooperative vs Non-cooperative (200 frames)")
    print("=" * 50)
    coop_metrics, noncoop_metrics, agent_diffs = run_coop_comparison(loader, n_frames=200)

    # =============================================
    # Save and summarize
    # =============================================

    # Main experiment summary
    print("\n" + "=" * 70)
    print("MAIN RESULTS (500 frames, real velocities)")
    print("=" * 70)
    method_names = list(methods.keys())
    print(f"\n{'Method':25s} {'Area':>8s} {'Tight%':>7s} {'MinTTC':>7s} {'MinHarm':>10s} {'Time':>7s}")
    print("-" * 70)
    rows = []
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        if not mm:
            continue
        a = np.mean([m.feasible_area for m in mm])
        t = np.mean([m.constraint_tightening_ratio for m in mm])
        ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh = sum(1 for m in mm if m.mode == "minimum_harm")
        tm = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:25s} {a:8.1f} {t:6.1%} {ttc:7.2f} {mh:>5d}/{len(mm):<4d} {tm:6.0f}ms")
        rows.append({"method": mname, "area": a, "tightening": t, "ttc": ttc,
                      "min_harm": mh, "total": len(mm), "time_ms": tm})

    # Cooperative comparison summary
    print("\n" + "=" * 70)
    print("COOPERATIVE vs NON-COOPERATIVE (200 frames, TRUE separation)")
    print("=" * 70)
    def _summarize(metrics, label):
        a = np.mean([m.feasible_area for m in metrics])
        t = np.mean([m.constraint_tightening_ratio for m in metrics])
        mh = sum(1 for m in metrics if m.mode == "minimum_harm")
        print(f"{label:25s} area={a:.1f}m², tight={t:.1%}, min-harm={mh}/{len(metrics)}")
        return {"area": a, "tightening": t, "min_harm": mh, "total": len(metrics)}

    coop_stats = _summarize(coop_metrics, "Cooperative (V2I)")
    noncoop_stats = _summarize(noncoop_metrics, "Non-cooperative (ego)")
    avg_coop_agents = np.mean([d["coop_agents"] for d in agent_diffs])
    avg_noncoop_agents = np.mean([d["noncoop_agents"] for d in agent_diffs])
    print(f"Avg agents: coop={avg_coop_agents:.1f}, noncoop={avg_noncoop_agents:.1f}, "
          f"diff={avg_coop_agents-avg_noncoop_agents:.1f}")

    # Save everything
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "DAIR-V2X-C (real velocities from consecutive frames)",
            "velocity_source": "computed from novatel_to_world position deltas, dt=0.1s",
            "n_frames_main": len(indices),
            "n_frames_coop": 200,
            "n_methods": len(methods),
            "method_names": method_names,
        },
        "main_results": rows,
        "cooperative_comparison": {
            "cooperative": coop_stats,
            "non_cooperative": noncoop_stats,
            "avg_coop_agents": avg_coop_agents,
            "avg_noncoop_agents": avg_noncoop_agents,
        },
        "metrics_main": [
            {k: (str(v) if isinstance(v, float) and (np.isinf(v) or np.isnan(v)) else v)
             for k, v in asdict(m).items()}
            for m in all_metrics
        ],
        "metrics_coop": [asdict(m) for m in coop_metrics],
        "metrics_noncoop": [asdict(m) for m in noncoop_metrics],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    # Human-readable summary
    summary = [
        "# Final Experiment Results — Real Velocities + Modern Baselines\n",
        f"Date: {datetime.now().isoformat()}",
        f"Dataset: DAIR-V2X-C, velocities from consecutive frame position deltas (dt=0.1s)\n",
        "",
        "## Main Comparison (500 frames × 9 methods)\n",
        "| Method | Avg Area (m²) | Constraint % | Min TTC | Min-Harm | Time |",
        "|--------|--------------|-------------|---------|----------|------|",
    ]
    for r in rows:
        summary.append(f"| {r['method']} | {r['area']:.1f} | {r['tightening']:.1%} | "
                        f"{r['ttc']:.2f}s | {r['min_harm']}/{r['total']} | {r['time_ms']:.0f}ms |")

    summary.extend([
        "",
        "## Cooperative vs Non-Cooperative (200 frames, TRUE label separation)\n",
        "| Setting | Avg Area | Constraint % | Min-Harm | Avg Agents |",
        "|---------|---------|-------------|----------|------------|",
        f"| Cooperative (V2I) | {coop_stats['area']:.1f} | {coop_stats['tightening']:.1%} | "
        f"{coop_stats['min_harm']}/{coop_stats['total']} | {avg_coop_agents:.1f} |",
        f"| Non-cooperative | {noncoop_stats['area']:.1f} | {noncoop_stats['tightening']:.1%} | "
        f"{noncoop_stats['min_harm']}/{noncoop_stats['total']} | {avg_noncoop_agents:.1f} |",
    ])
    (output_dir / "summary.md").write_text("\n".join(summary))
    (output_dir / "reproduction_info.json").write_text(json.dumps({
        "command": "python experiments/run_final_experiments.py",
        "env": "coop-safety",
        "data": FULL_DATA_DIR,
    }, indent=2))

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()

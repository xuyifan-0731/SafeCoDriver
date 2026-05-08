"""Run experiments on DAIR-V2X-C FULL dataset — all 6601 frames.

Core methods only (no redundant ablations) to keep runtime tractable.
Methods: NoConstraint, RSS-Only, CBF-Based, Ours-Full, Ours-NoRiskEvents, Ours-RiskMapOnly

Usage:
    cd /raid/xuyifan/jiqiuyu
    conda activate coop-safety
    python experiments/run_dairv2x_all_frames.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.dairv2x_loader import DAIRv2xLoader
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario, ScenarioMetrics

FULL_DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"


def get_core_methods() -> dict:
    """Only the essential methods for the full-scale run."""
    from coop_safety.interface import SafetyConstraintModule
    from experiments.methods import NoConstraint, RSSOnly, CBFBased, AblationMethod

    methods = {}
    methods["NoConstraint"] = NoConstraint()
    methods["RSS-Only"] = RSSOnly()
    methods["CBF-Based"] = CBFBased()

    full = SafetyConstraintModule()
    full.name = "Ours-Full"
    methods["Ours-Full"] = full

    # Key ablations only
    methods["Ours-NoRiskEvents"] = AblationMethod(
        "Ours-NoRiskEvents", {"disable_risk_events": True, "min_probability": 999})
    methods["Ours-RiskMapOnly"] = AblationMethod(
        "Ours-RiskMapOnly", {"risk_map_only": True, "ttc_warning": 0.0, "min_probability": 999})

    return methods


def main():
    output_dir = Path(__file__).parent / "results" / f"dairv2x_ALL_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: DAIR-V2X-C ALL FRAMES (6601)")
    print("=" * 70)

    loader = DAIRv2xLoader(FULL_DATA_DIR, ego_speed=10.0)
    n_total = len(loader)
    print(f"Total frames: {n_total}")

    methods = get_core_methods()
    method_names = list(methods.keys())
    print(f"Methods ({len(methods)}): {method_names}")

    # Run all frames
    all_metrics = []
    t_start = time.time()
    errors = 0

    for idx in range(n_total):
        if idx % 200 == 0:
            elapsed = time.time() - t_start
            rate = idx / max(elapsed, 0.1)
            eta = (n_total - idx) / max(rate, 0.01)
            print(f"\n[{idx}/{n_total}] elapsed={elapsed:.0f}s, rate={rate:.1f} frames/s, ETA={eta:.0f}s")

        try:
            perception = loader.load_frame(idx)
        except Exception as e:
            errors += 1
            continue

        n_agents = len(perception.agents)
        difficulty = "easy" if n_agents <= 5 else ("medium" if n_agents <= 15 else "hard")
        config = ScenarioConfig(
            name=f"frame_{loader.frames[idx]}",
            description=f"{n_agents} agents",
            seed=42 + idx,
            difficulty=difficulty,
        )

        for mname, method in methods.items():
            metrics = evaluate_method_on_scenario(method, perception, config)
            all_metrics.append(metrics)

    elapsed_total = time.time() - t_start
    print(f"\nDone. {n_total} frames × {len(methods)} methods = {len(all_metrics)} experiments")
    print(f"Total time: {elapsed_total:.0f}s ({elapsed_total/n_total:.2f}s per frame)")
    print(f"Errors: {errors}")

    # Save results
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "DAIR-V2X-C FULL (all frames)",
            "data_dir": FULL_DATA_DIR,
            "n_frames": n_total,
            "n_methods": len(methods),
            "method_names": method_names,
            "total_experiments": len(all_metrics),
            "elapsed_seconds": elapsed_total,
            "errors": errors,
        },
        "metrics": [
            {k: (str(v) if isinstance(v, float) and (np.isinf(v) or np.isnan(v)) else v)
             for k, v in asdict(m).items()}
            for m in all_metrics
        ],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    # Aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS (ALL 6601 FRAMES)")
    print("=" * 70)
    print(f"\n{'Method':25s} {'Area':>10s} {'Tight%':>8s} {'MinTTC':>8s} {'MinHarm':>8s} {'Time':>8s}")
    print("-" * 70)
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        if not mm:
            continue
        avg_area = np.mean([m.feasible_area for m in mm])
        avg_tight = np.mean([m.constraint_tightening_ratio for m in mm])
        avg_ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh_count = sum(1 for m in mm if m.mode == "minimum_harm")
        avg_time = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:25s} {avg_area:10.1f} {avg_tight:7.1%} {avg_ttc:8.2f} "
              f"{mh_count:>5d}/{len(mm):<3d} {avg_time:7.0f}ms")

    # Save summary
    summary_lines = ["# DAIR-V2X-C Full Dataset Results (ALL 6601 Frames)\n"]
    summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    summary_lines.append(f"Total experiments: {len(all_metrics)}\n")
    summary_lines.append(f"| Method | Avg Area (m²) | Constraint % | Avg Min TTC | Min-Harm | Avg Time |")
    summary_lines.append(f"|--------|--------------|-------------|------------|----------|----------|")
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        if not mm:
            continue
        avg_area = np.mean([m.feasible_area for m in mm])
        avg_tight = np.mean([m.constraint_tightening_ratio for m in mm])
        avg_ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh_count = sum(1 for m in mm if m.mode == "minimum_harm")
        avg_time = np.mean([m.compute_time_ms for m in mm])
        summary_lines.append(
            f"| {mname} | {avg_area:.1f} | {avg_tight:.1%} | {avg_ttc:.2f}s | "
            f"{mh_count}/{len(mm)} | {avg_time:.0f}ms |")

    (output_dir / "summary.md").write_text("\n".join(summary_lines))

    # Reproduction info
    repro = {
        "command": f"cd /raid/xuyifan/jiqiuyu && /raid/xuyifan/miniconda3/envs/coop-safety/bin/python experiments/run_dairv2x_all_frames.py",
        "conda_env": "coop-safety",
        "data_dir": FULL_DATA_DIR,
        "n_frames": n_total,
    }
    (output_dir / "reproduction_info.json").write_text(json.dumps(repro, indent=2))

    return output_dir


if __name__ == "__main__":
    main()

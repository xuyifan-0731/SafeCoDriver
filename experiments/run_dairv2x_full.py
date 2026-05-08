"""Run experiments on DAIR-V2X-C full dataset (6601 frames).

Samples 100 frames evenly for tractable evaluation.
All methods × all frames, with full reproducibility.

Usage:
    cd /raid/xuyifan/jiqiuyu
    conda activate coop-safety
    python experiments/run_dairv2x_full.py
"""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.dairv2x_loader import DAIRv2xLoader, get_dairv2x_scenarios
from experiments.scenarios import ScenarioConfig
from experiments.methods import get_all_methods
from experiments.run_experiments import (
    evaluate_method_on_scenario,
    generate_summary_table,
    ScenarioMetrics,
)

# Configuration
FULL_DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
EXAMPLE_DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Example/example-cooperative-vehicle-infrastructure"
N_FRAMES = 100  # Sample 100 frames from 6601
SEED = 42


def main():
    output_dir = Path(__file__).parent / "results" / f"dairv2x_full_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Safety Constraints on DAIR-V2X-C Full Dataset")
    print("=" * 70)

    # Try full dataset first, fallback to example
    data_dir = FULL_DATA_DIR
    if not Path(data_dir).exists():
        print(f"[WARN] Full dataset not found at {data_dir}, using example data")
        data_dir = EXAMPLE_DATA_DIR

    scenarios = get_dairv2x_scenarios(
        data_dir=data_dir,
        n_frames=N_FRAMES,
        seed=SEED,
    )
    if not scenarios:
        print("ERROR: No data found")
        return

    methods = get_all_methods()

    print(f"\nDataset: {data_dir}")
    print(f"Frames sampled: {len(scenarios)} (seed={SEED})")
    print(f"Methods: {len(methods)}")
    for name in methods:
        print(f"  - {name}")

    # Difficulty distribution
    diff_counts = {}
    for _, c in scenarios:
        diff_counts[c.difficulty] = diff_counts.get(c.difficulty, 0) + 1
    print(f"Difficulty distribution: {diff_counts}")

    # Run experiments
    all_metrics = []
    t_start = time.time()

    for i, (perception, config) in enumerate(scenarios):
        n_agents = len(perception.agents)
        print(f"\n[{i+1}/{len(scenarios)}] {config.name} ({config.difficulty}, {n_agents} agents)")

        for method_name, method in methods.items():
            metrics = evaluate_method_on_scenario(method, perception, config)
            status = "OK" if not metrics.error else f"ERR"
            print(f"  {method_name:20s} area={metrics.feasible_area:8.1f} "
                  f"ttc={metrics.min_ttc:6.2f} mode={metrics.mode:13s} "
                  f"time={metrics.compute_time_ms:5.0f}ms [{status}]")
            all_metrics.append(metrics)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/len(scenarios):.1f}s per frame)")

    # Save results
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "DAIR-V2X-C (real-world cooperative perception)",
            "data_dir": data_dir,
            "n_frames_total": len(DAIRv2xLoader(data_dir)),
            "n_frames_sampled": len(scenarios),
            "n_methods": len(methods),
            "method_names": list(methods.keys()),
            "seed": SEED,
            "total_experiments": len(all_metrics),
            "elapsed_seconds": elapsed,
            "note": "Real-world V2I cooperative data. Ego velocity estimated at 10 m/s. "
                    "Lane geometry synthetic. Frame selection deterministic (seed=42).",
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
    print(f"\nResults: {results_path}")

    summary = generate_summary_table(all_metrics)
    summary_path = output_dir / "summary_table.md"
    summary_path.write_text(summary)
    print(f"Summary: {summary_path}")

    # Reproduction info
    repro = {
        "command": f"cd /raid/xuyifan/jiqiuyu && "
                   f"/raid/xuyifan/miniconda3/envs/coop-safety/bin/python "
                   f"experiments/run_dairv2x_full.py",
        "conda_env": "coop-safety",
        "data_dir": data_dir,
        "n_frames": N_FRAMES,
        "seed": SEED,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    (output_dir / "reproduction_info.json").write_text(json.dumps(repro, indent=2))

    # Print aggregate summary
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    method_names = sorted(set(m.method_name for m in all_metrics))
    print(f"\n{'Method':25s} {'Area':>10s} {'Tight%':>8s} {'MinTTC':>8s} {'MinHarm':>8s} {'Time':>8s}")
    print("-" * 70)
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        avg_area = np.mean([m.feasible_area for m in mm])
        avg_tight = np.mean([m.constraint_tightening_ratio for m in mm])
        avg_ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh_count = sum(1 for m in mm if m.mode == "minimum_harm")
        avg_time = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:25s} {avg_area:10.1f} {avg_tight:7.1%} {avg_ttc:8.2f} {mh_count:8d} {avg_time:7.0f}ms")

    return output_dir


if __name__ == "__main__":
    main()

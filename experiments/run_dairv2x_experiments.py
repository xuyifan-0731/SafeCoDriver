"""Run experiments on DAIR-V2X real-world data.

Uses the same evaluation framework as synthetic experiments
but with real cooperative perception data from DAIR-V2X dataset.

Usage:
    cd /raid/xuyifan/jiqiuyu
    conda activate coop-safety
    python experiments/run_dairv2x_experiments.py

Output:
    experiments/results/dairv2x_YYYY-MM-DD_HHMMSS/
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.dairv2x_loader import get_dairv2x_scenarios
from experiments.methods import get_all_methods
from experiments.run_experiments import (
    evaluate_method_on_scenario,
    generate_summary_table,
    ScenarioMetrics,
)


def main():
    output_dir = Path(__file__).parent / "results" / f"dairv2x_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Safety Constraints on DAIR-V2X Real-World Data")
    print("=" * 70)

    # Load real data scenarios
    scenarios = get_dairv2x_scenarios(n_frames=10, seed=42)
    if not scenarios:
        print("ERROR: No DAIR-V2X data found. See 数据集下载指引.md")
        return

    methods = get_all_methods()

    print(f"\nScenarios: {len(scenarios)} real-world frames")
    for p, c in scenarios:
        print(f"  - {c.name} ({c.difficulty}): {c.description}")
    print(f"\nMethods: {len(methods)}")

    # Run experiments
    all_metrics = []
    for perception, config in scenarios:
        print(f"\n--- {config.name} ({config.difficulty}, {len(perception.agents)} agents) ---")
        for method_name, method in methods.items():
            print(f"  {method_name}...", end=" ", flush=True)
            metrics = evaluate_method_on_scenario(method, perception, config)
            status = "OK" if not metrics.error else f"ERR: {metrics.error[:50]}"
            print(f"area={metrics.feasible_area:.1f}, ttc={metrics.min_ttc:.2f}, "
                  f"mode={metrics.mode}, time={metrics.compute_time_ms:.0f}ms [{status}]")
            all_metrics.append(metrics)

    # Save results
    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "dataset": "DAIR-V2X-C (real-world cooperative perception)",
            "data_source": "DAIR-V2X-C-Example (CVPR 2022)",
            "n_scenarios": len(scenarios),
            "n_methods": len(methods),
            "note": "Real-world data. Ego velocity estimated (not from ground truth). "
                    "Lane geometry is synthetic (DAIR-V2X doesn't provide lane annotations).",
        },
        "metrics": [
            {k: (v if not isinstance(v, float) or not np.isinf(v) else "inf")
             for k, v in m.__dict__.items()}
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
                   f"experiments/run_dairv2x_experiments.py",
        "conda_env": "coop-safety",
        "data_path": "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Example/"
                     "example-cooperative-vehicle-infrastructure",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "note": "Deterministic frame selection (seed=42, evenly spaced). "
                "Results reproducible given same data and code.",
    }
    (output_dir / "reproduction_info.json").write_text(json.dumps(repro, indent=2))

    print("\n" + summary)
    return output_dir


if __name__ == "__main__":
    main()

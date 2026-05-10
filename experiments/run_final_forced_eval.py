"""Final SUMO + FA experiment combining all findings.

- Forced-conflict scenarios (conflict types 1 and 3 reliably produce collisions)
- Multiple seeds for stability
- Reports per-method: collision count, severity reduction, detection lead time
- Uses best Hybrid variant: thr=0.40 with optional AND-fusion
"""
from __future__ import annotations
import sys, os, math, time, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety
from experiments.run_forced_conflict_and_fa import (
    generate_forced_conflict, run_sumo_forced, HybridWithGeometricAND)

SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'


def main():
    print("=" * 70)
    print("  Final Forced-Conflict SUMO Eval (260510)")
    print("=" * 70)
    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    base_hyb = HybridSafetyConstraint(
        detector_model=v1, base_margin_visible=2.5,
        base_margin_invisible=4.0, detection_threshold=0.40)

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid (thr=0.40)": base_hyb,
        "Ours-Hybrid+AND": HybridWithGeometricAND(base_hyb),
    }

    output_dir = SCENARIO_DIR / 'networks'
    # Use only conflict types 1 (head-on) and 3 (rear-end) which reliably produce collisions
    conflict_types_to_run = [1, 3]
    seeds = [42, 123, 789]
    networks = ['t_junction_1lane', 'crossroads_1lane']

    all_results = {n: [] for n in methods}

    for net_name in networks:
        net_file = output_dir / f"{net_name}.net.xml"
        if not net_file.exists():
            continue
        for ct in conflict_types_to_run:
            for seed in seeds:
                # Encode (conflict type, seed) into scenario_id by ct + seed_offset*4
                # Use 5 different scenario_ids per (ct, seed) for variety
                for sid_offset in range(5):
                    actual_sid = ct + sid_offset * 4
                    rou_file = generate_forced_conflict(
                        net_file, actual_sid, net_name, seed=seed)
                    for mname, method in methods.items():
                        # Reset stateful methods
                        if hasattr(method, 'recent_probs'):
                            method.recent_probs.clear()
                        if hasattr(method, 'recent_geom'):
                            method.recent_geom.clear()
                        r = run_sumo_forced(net_file, rou_file, method, sumo_seed=seed)
                        all_results[mname].append({
                            'net': net_name, 'ct': ct, 'seed': seed,
                            'sid': actual_sid, 'result': r,
                        })
        # Progress
        n_total = sum(1 for nm in [net_name] for ct in conflict_types_to_run
                      for s in seeds for o in range(5))
        for mname in methods:
            recent = [x for x in all_results[mname] if x['net'] == net_name]
            tc = sum(x['result'].unique_ego_coll for x in recent)
            print(f"  {net_name} {mname:25s}: {tc} coll / {len(recent)} runs")

    # Final results
    print("\n" + "=" * 70)
    print("  RESULTS: Forced-Conflict Eval")
    print("=" * 70)
    print(f"  {'Method':25s} {'Coll':>6s} {'Sev':>6s} {'Coll-by-CT':>15s} {'Early':>6s}")
    print("  " + "-" * 70)
    for mname in methods:
        results = all_results[mname]
        n_runs = len(results)
        tc = sum(x['result'].unique_ego_coll for x in results)
        sevs = [s for x in results for s in x['result'].severities]
        ct_breakdown = {1: 0, 3: 0}
        for x in results:
            ct_breakdown[x['ct']] = ct_breakdown.get(x['ct'], 0) + x['result'].unique_ego_coll
        # Early warning: if collision happened, how many frames before?
        early_frames = []
        for x in results:
            r = x['result']
            if r.unique_ego_coll > 0 and r.first_warning >= 0 and r.first_collision >= 0:
                early_frames.append(r.first_collision - r.first_warning)
        avg_early = np.mean(early_frames) if early_frames else 0
        print(f"  {mname:25s} {tc:3d}/{n_runs:3d} {np.mean(sevs) if sevs else 0:5.2f} "
              f"  CT1={ct_breakdown[1]:2d} CT3={ct_breakdown[3]:2d}  {avg_early:5.1f}")


if __name__ == "__main__":
    main()

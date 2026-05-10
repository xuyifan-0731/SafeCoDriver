"""Streamlined forced-conflict SUMO: focus on key methods only.

Skip Shapely-heavy methods (RSS/APF/CBF) which take >100s each.
Focus on: NoConstraint, UniE2EV2X, MAP, Ours-Hybrid, Ours+AND.
"""
from __future__ import annotations
import sys, os, math, time, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety
from experiments.run_forced_conflict_and_fa import (
    generate_forced_conflict, run_sumo_forced, HybridWithGeometricAND)

SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'


def main():
    print("=" * 70)
    print("  Streamlined Forced-Conflict (260510)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    base_hyb = HybridSafetyConstraint(
        detector_model=v1, base_margin_visible=2.5,
        base_margin_invisible=4.0, detection_threshold=0.40)

    # Skip slow Shapely-based: only fast methods
    methods = {
        "NoConstraint": None,
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "Ours-Hybrid (thr=0.40)": base_hyb,
        "Ours-Hybrid+AND": HybridWithGeometricAND(base_hyb),
    }

    output_dir = SCENARIO_DIR / 'networks'
    networks = ['t_junction_1lane', 'crossroads_1lane']
    conflict_types = [1, 3]  # head-on and rear-end (reliable conflicts)
    seeds = [42, 123, 789]
    n_sids_per_ct = 4  # 4 different scenarios per (ct, seed)

    all_results = {n: [] for n in methods}
    t0 = time.time()

    for net_name in networks:
        net_file = output_dir / f"{net_name}.net.xml"
        if not net_file.exists():
            print(f"  Skip {net_name}: no file"); continue
        print(f"\n  --- {net_name} ---")
        for ct in conflict_types:
            for seed in seeds:
                for sid_off in range(n_sids_per_ct):
                    actual_sid = ct + sid_off * 4
                    rou_file = generate_forced_conflict(
                        net_file, actual_sid, net_name, seed=seed)
                    for mname, method in methods.items():
                        if hasattr(method, 'recent_probs'):
                            method.recent_probs.clear()
                        r = run_sumo_forced(net_file, rou_file, method, sumo_seed=seed)
                        all_results[mname].append({
                            'net': net_name, 'ct': ct, 'seed': seed, 'r': r,
                        })
        # Per-network progress
        for mname in methods:
            recent = [x for x in all_results[mname] if x['net'] == net_name]
            tc = sum(x['r'].unique_ego_coll for x in recent)
            sevs = [s for x in recent for s in x['r'].severities]
            avg_sev = np.mean(sevs) if sevs else 0
            print(f"    {mname:25s}: {tc}/{len(recent)} coll, sev={avg_sev:.2f}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")

    # Aggregate
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  {'Method':25s} {'Coll/Total':>11s} {'Sev':>6s} "
          f"{'CT1':>5s} {'CT3':>5s} {'EarlyB4Coll':>11s}")
    print("  " + "-" * 65)
    for mname in methods:
        results = all_results[mname]
        tc = sum(x['r'].unique_ego_coll for x in results)
        sevs = [s for x in results for s in x['r'].severities]
        ct1 = sum(x['r'].unique_ego_coll for x in results if x['ct'] == 1)
        ct3 = sum(x['r'].unique_ego_coll for x in results if x['ct'] == 3)
        # Frames between first warning and first collision (only for collisions)
        early = []
        for x in results:
            r = x['r']
            if r.unique_ego_coll > 0 and r.first_warning >= 0 and r.first_collision >= 0:
                early.append(max(0, r.first_collision - r.first_warning))
        avg_e = np.mean(early) if early else 0
        print(f"  {mname:25s} {tc:5d}/{len(results):5d} "
              f"{np.mean(sevs) if sevs else 0:5.2f} {ct1:4d} {ct3:4d}  {avg_e:10.1f}")


if __name__ == "__main__":
    main()

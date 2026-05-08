"""Fast SUMO sanity check with corrected methods.

Reduced to 5 scenarios × 1 seed × 7 methods to fit in reasonable time.
Verifies that the corrected baseline mode signaling doesn't break SUMO results.
"""
from __future__ import annotations
import sys, os, math, time, random, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Reuse infrastructure from run_corrected_eval
from experiments.run_corrected_eval import (
    get_or_create_network, generate_routes, run_sumo, SCENARIO_DIR,
    NEW_V1_THRESHOLD)

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety


def main():
    print("=" * 70)
    print("  SUMO Quick Sanity Check (corrected methods, 5 scenarios × 1 seed)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "APF": RiskPotentialField(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=NEW_V1_THRESHOLD),
    }

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_sc = 5
    seed = 42
    stypes = [("t_junction", n_sc), ("crossroads", n_sc)]
    all_results = {n: {s: [] for s, _ in stypes} for n in methods}
    t0 = time.time()

    for stype, _ in stypes:
        print(f"\n  --- {stype} ({n_sc} scenarios) ---")
        net_file = get_or_create_network(stype, output_dir)
        for sid in range(n_sc):
            rou_file = generate_routes(net_file, stype, sid, n_veh=10, seed=seed)
            for mname, method in methods.items():
                t_method = time.time()
                r = run_sumo(net_file, rou_file, method, sumo_seed=seed)
                all_results[mname][stype].append(r)
                elapsed = time.time() - t_method
                if elapsed > 60:
                    print(f"    [{sid}/{n_sc}] {mname}: {elapsed:.0f}s, "
                          f"coll={r.unique_ego_coll}")

            # Quick progress
            ro = all_results["Ours-Hybrid"][stype][-1]
            rn = all_results["NoConstraint"][stype][-1]
            print(f"    [{sid}] NoCon={rn.unique_ego_coll}, Ours={ro.unique_ego_coll}, "
                  f"Ours_2nd={ro.secondary_coll}")

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    for stype, _ in stypes:
        print(f"\n  [{stype}]")
        print(f"  {'Method':22s} {'Coll':>6s} {'2nd':>5s} {'2ndR':>6s} {'Sev':>6s} {'WPC%':>6s}")
        print("  " + "-" * 55)
        for mname in methods:
            results = all_results[mname][stype]
            tc = sum(r.unique_ego_coll for r in results)
            ts = sum(r.secondary_coll for r in results)
            sevs = [s for r in results for s in r.severities]
            tw = sum(r.wp_total for r in results)
            twc = sum(r.wp_coll for r in results)
            print(f"  {mname:22s} {tc:6d} {ts:5d} {ts/max(tc,1):5.0%} "
                  f"{np.mean(sevs) if sevs else 0:5.1f} {twc/max(tw,1):5.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""Comprehensive final evaluation (260511) — paper-ready table.

All methods × all variants × all metrics on:
  - DeepAccident val (22 unseen scenarios)
  - SUMO blind-spot (3 scenarios × 3 seeds × 4 sids)

Skips slow methods (APF/RSS Shapely-heavy) for SUMO when not needed,
runs all on DA.

Output: complete table with full metrics for each row.
"""
from __future__ import annotations
import sys, os, math, time, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND
from experiments.run_blindspot_unified import (
    get_or_create_network, gen_BS1_uturn, gen_BS3_blindcorner,
    gen_BS4_hidden_merge, run_scenario)
from experiments.run_deepaccident_unified_metrics import eval_method
from experiments.deepaccident_loader import DeepAccidentLoader


def eval_sumo(method, scenarios, seeds, n_sids=4, ego_uses_coop=True):
    n_runs = 0; coll_runs = 0; ego_coll = 0; sec = 0
    sevs = []
    n_dangerous = 0; n_warned_dang = 0
    n_safe = 0; n_warned_safe = 0
    early_warns = []
    for sname, net_file, gen_fn in scenarios:
        for seed in seeds:
            for sid in range(n_sids):
                rou = gen_fn(net_file, sid, seed)
                if hasattr(method, 'recent_probs'):
                    method.recent_probs.clear()
                r = run_scenario(net_file, rou, method,
                                 ego_uses_coop=ego_uses_coop, sumo_seed=seed)
                n_runs += 1
                ego_coll += r.unique_ego_coll
                sec += r.secondary_coll
                sevs.extend(r.severities)
                n_dangerous += r.n_dangerous_frames
                n_warned_dang += r.n_warned_dangerous
                n_safe += r.n_safe_frames
                n_warned_safe += r.n_warned_safe
                if r.is_collision_scenario:
                    coll_runs += 1
                    if r.first_warning >= 0 and r.first_warning <= r.first_collision:
                        early_warns.append(r.first_collision - r.first_warning)
    return {
        'CollRate': coll_runs / max(n_runs, 1),
        'TotColl': ego_coll,
        'Sec': sec,
        'Sev': np.mean(sevs) if sevs else 0,
        'Det(f)_dang': n_warned_dang / max(n_dangerous, 1),
        'FA(f)': n_warned_safe / max(n_safe, 1),
        'Early': np.mean(early_warns) if early_warns else 0,
    }


def main():
    print("=" * 70)
    print("  COMPREHENSIVE FINAL EVAL (260511)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc', 0):.4f} (scenario-split, no leakage)")
    v1 = CollisionPredictionNetwork(); v1.load_state_dict(ckpt['model']); v1.eval()
    val_idx = ckpt.get('val_scenario_idx', [])
    print(f"  Val scenarios: {len(val_idx)}")

    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5,
                       base_margin_invisible=4.0)
    cross_1l = get_or_create_network('cross_1lane')
    cross_2l = get_or_create_network('cross_2lane')
    sumo_scenarios = [
        ('BS1', cross_1l, gen_BS1_uturn),
        ('BS3', cross_1l, gen_BS3_blindcorner),
        ('BS4', cross_2l, gen_BS4_hidden_merge),
    ]
    seeds = [42, 123, 789]

    methods = [
        # Baselines (no V2X)
        ('NoCon-egoonly', lambda: None, False),
        # Baselines (with V2X coop)
        ('NoCon-coop', lambda: None, True),
        ('RSS-coop', lambda: RSSOnly(), True),
        ('APF-coop', lambda: RiskPotentialField(), True),
        ('UniE2EV2X-coop', lambda: UniE2EV2XSafety(safety_threshold=3.0), True),
        ('MAP-coop', lambda: MAPSafety(min_clearance=0.5), True),
        ('RiskMM-coop', lambda: RiskMMSafety(v_max=20.0), True),
        # Hybrid variants (P0+P1 in code)
        ('Hybrid-thr0.20', lambda: HybridSafetyConstraint(detection_threshold=0.20, **base_kwargs), True),
        ('Hybrid-thr0.25', lambda: HybridSafetyConstraint(detection_threshold=0.25, **base_kwargs), True),
        ('Hybrid-thr0.30', lambda: HybridSafetyConstraint(detection_threshold=0.30, **base_kwargs), True),
        ('Hybrid-thr0.35', lambda: HybridSafetyConstraint(detection_threshold=0.35, **base_kwargs), True),
        ('Hybrid-thr0.40', lambda: HybridSafetyConstraint(detection_threshold=0.40, **base_kwargs), True),
        ('Hybrid-thr0.50', lambda: HybridSafetyConstraint(detection_threshold=0.50, **base_kwargs), True),
        # Hybrid+AND (P0+P1)
        ('Hybrid+AND-thr0.25',
         lambda: HybridWithGeometricAND(HybridSafetyConstraint(detection_threshold=0.25, **base_kwargs)),
         True),
        ('Hybrid+AND-thr0.30',
         lambda: HybridWithGeometricAND(HybridSafetyConstraint(detection_threshold=0.30, **base_kwargs)),
         True),
        ('Hybrid+AND-thr0.40',
         lambda: HybridWithGeometricAND(HybridSafetyConstraint(detection_threshold=0.40, **base_kwargs)),
         True),
    ]

    loader = DeepAccidentLoader(split='all')

    # Combined results table
    sumo_results = {}
    da_results = {}

    print("\n  Running SUMO blind-spot eval (this will take ~25 min)...")
    print(f"  {'Method':25s} {'CollRate':>8s} {'TotC':>5s} {'Sec':>4s} "
          f"{'Sev':>5s} {'Early':>6s} {'FA(f)':>6s}")
    print("  " + "-" * 75)
    t0 = time.time()
    for name, factory, uses_coop in methods:
        method = factory()
        r = eval_sumo(method, sumo_scenarios, seeds, ego_uses_coop=uses_coop)
        sumo_results[name] = r
        print(f"  {name:25s} {r['CollRate']:7.0%} {r['TotColl']:5d} {r['Sec']:4d} "
              f"{r['Sev']:5.2f} {r['Early']:5.1f} {r['FA(f)']:5.1%}")
    print(f"\n  SUMO eval done in {time.time()-t0:.0f}s")

    print("\n  Running DeepAccident eval...")
    print(f"  {'Method':25s} {'Det(s)':>7s} {'Det(f)':>7s} {'Early':>6s} "
          f"{'FA(s)':>6s} {'FA(f)':>6s} {'WPC%':>6s} {'Mod%':>5s}")
    print("  " + "-" * 80)
    t0 = time.time()
    for name, factory, uses_coop in methods:
        method = factory()
        r = eval_method(method, val_idx, loader, ego_uses_coop=uses_coop)
        da_results[name] = r
        print(f"  {name:25s} {r['Det(s)']:6.0%} {r['Det(f)']:6.0%} "
              f"{r['Early']:5.1f} {r['FA(s)']:5.0%} {r['FA(f)']:5.0%} "
              f"{r['WPColl']:5.1%} {r['Mod']:4.0%}")
    print(f"\n  DA eval done in {time.time()-t0:.0f}s")

    # Combined table
    print("\n" + "=" * 110)
    print("  FULL COMBINED TABLE")
    print("=" * 110)
    print(f"  {'Method':25s} | "
          f"{'SUMO':^28s} | "
          f"{'DeepAccident':^48s}")
    print(f"  {'':25s} | "
          f"{'CollR':>5s} {'TotC':>5s} {'Sec':>4s} {'Sev':>5s} "
          f"| {'Det(s)':>6s} {'Det(f)':>6s} {'Early':>6s} {'FA(s)':>5s} {'FA(f)':>5s} {'WPC%':>5s} {'Mod%':>4s}")
    print('  ' + '-' * 105)
    for name, _, _ in methods:
        s = sumo_results[name]
        d = da_results[name]
        print(f"  {name:25s} | {s['CollRate']:4.0%} {s['TotColl']:5d} {s['Sec']:4d} {s['Sev']:5.2f} "
              f"| {d['Det(s)']:5.0%} {d['Det(f)']:5.0%} {d['Early']:5.1f} {d['FA(s)']:4.0%} "
              f"{d['FA(f)']:4.0%} {d['WPColl']:4.1%} {d['Mod']:3.0%}")

    # Per-scenario SUMO breakdown
    print("\n" + "=" * 70)
    print("  SUMO PER-SCENARIO BREAKDOWN (CollRate per scenario)")
    print("=" * 70)
    print(f"  {'Method':25s} {'BS1':>5s} {'BS3':>5s} {'BS4':>5s}  {'Total':>6s}")
    print('  ' + '-' * 50)
    for name, factory, uses_coop in methods:
        method = factory()
        bs_results = {}
        for sname, net_file, gen_fn in sumo_scenarios:
            r = eval_sumo(method, [(sname, net_file, gen_fn)], seeds,
                          ego_uses_coop=uses_coop)
            bs_results[sname] = r['CollRate']
            method = factory()  # reset for next scenario
        total = sumo_results[name]['CollRate']
        print(f"  {name:25s} {bs_results.get('BS1', 0):4.0%} {bs_results.get('BS3', 0):4.0%} "
              f"{bs_results.get('BS4', 0):4.0%}  {total:5.0%}")

    # Save results to file
    import json
    output = {
        'sumo': {k: v for k, v in sumo_results.items()},
        'deepaccident': {k: v for k, v in da_results.items()},
    }
    out_path = Path(__file__).parent / 'comprehensive_final_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved JSON: {out_path}")


if __name__ == "__main__":
    main()

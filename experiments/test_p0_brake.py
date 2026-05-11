"""P0 improvement test: explicit target_speed_factor in Hybrid.

Compares baseline Hybrid (no factor) vs P0 (with factor) on:
- SUMO blind-spot: 3 scenarios × 3 seeds × 4 sids = 36 runs (per method)
- DeepAccident val (22 scenarios)

Checks: SUMO CollRate ↓, DA WPColl% unchanged, Det/FA monitored.
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
from experiments.run_blindspot_unified import (
    get_or_create_network, gen_BS1_uturn, gen_BS3_blindcorner,
    gen_BS4_hidden_merge, run_scenario, FullMetrics)
from experiments.run_deepaccident_unified_metrics import eval_method
from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND


def test_sumo(method, scenarios, seeds, n_sids=4, ego_uses_coop=True):
    """Returns dict of aggregated SUMO metrics."""
    n_runs = 0; coll_runs = 0; ego_coll = 0; sec = 0
    sevs = []; warning_frames = []; coll_frames = []
    n_dangerous = 0; n_warned_dang = 0
    n_safe = 0; n_warned_safe = 0
    early_warns = []
    for sname, net_file, gen_fn in scenarios:
        for seed in seeds:
            for sid in range(n_sids):
                rou = gen_fn(net_file, sid, seed)
                # Reset stateful
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
        '2nd': sec,
        'Sev': np.mean(sevs) if sevs else 0,
        'Det(f)': n_warned_dang / max(n_dangerous, 1),
        'FA(f)': n_warned_safe / max(n_safe, 1),
        'Early': np.mean(early_warns) if early_warns else 0,
        'n_runs': n_runs,
    }


def main():
    print("=" * 70)
    print("  P0 Test: Explicit target_speed_factor in Hybrid (260511)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    v1 = CollisionPredictionNetwork(); v1.load_state_dict(ckpt['model']); v1.eval()
    val_idx = ckpt.get('val_scenario_idx', [])

    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5,
                       base_margin_invisible=4.0)
    cross_1l = get_or_create_network('cross_1lane')
    cross_2l = get_or_create_network('cross_2lane')
    scenarios = [
        ('BS1', cross_1l, gen_BS1_uturn),
        ('BS3', cross_1l, gen_BS3_blindcorner),
        ('BS4', cross_2l, gen_BS4_hidden_merge),
    ]
    seeds = [42, 123, 789]
    loader = DeepAccidentLoader(split='all')

    methods = [
        ('Hybrid-thr0.25 (P0)', lambda: HybridSafetyConstraint(detection_threshold=0.25, **base_kwargs)),
        ('Hybrid-thr0.30 (P0)', lambda: HybridSafetyConstraint(detection_threshold=0.30, **base_kwargs)),
        ('Hybrid-thr0.40 (P0)', lambda: HybridSafetyConstraint(detection_threshold=0.40, **base_kwargs)),
        ('Hybrid+AND-thr0.25 (P0)', lambda: HybridWithGeometricAND(
            HybridSafetyConstraint(detection_threshold=0.25, **base_kwargs))),
        ('Hybrid+AND-thr0.40 (P0)', lambda: HybridWithGeometricAND(
            HybridSafetyConstraint(detection_threshold=0.40, **base_kwargs))),
    ]

    # SUMO test
    print("\n  SUMO Blind-Spot Results (P0):")
    print(f"  {'Method':30s} {'CollRate':>8s} {'TotColl':>7s} {'Sev':>5s} "
          f"{'Det(f)':>6s} {'FA(f)':>6s} {'Early':>6s}")
    print("  " + "-" * 75)
    sumo_results = {}
    for name, factory in methods:
        method = factory()
        r = test_sumo(method, scenarios, seeds)
        sumo_results[name] = r
        print(f"  {name:30s} {r['CollRate']:7.0%} {r['TotColl']:6d} "
              f"{r['Sev']:5.1f} {r['Det(f)']:5.0%} {r['FA(f)']:5.0%} {r['Early']:5.1f}")

    # DA test (just check WPColl/Det/FA didn't regress)
    print("\n  DeepAccident (val 22) Results (P0):")
    print(f"  {'Method':30s} {'Det(s)':>7s} {'FA(s)':>6s} {'FA(f)':>6s} "
          f"{'WPC%':>5s} {'Mod%':>5s} {'Early':>6s}")
    print("  " + "-" * 75)
    da_results = {}
    for name, factory in methods:
        method = factory()
        r = eval_method(method, val_idx, loader, ego_uses_coop=True)
        da_results[name] = r
        print(f"  {name:30s} {r['Det(s)']:6.0%} {r['FA(s)']:5.0%} {r['FA(f)']:5.0%} "
              f"{r['WPColl']:5.1%} {r['Mod']:4.0%} {r['Early']:5.1f}")

    # Compare with P0-baseline (RSS, NoCon)
    print("\n  Reference: SUMO baselines from previous run")
    print("  Method                CollRate  Sev   FA(f)")
    print("  ---------------------------------------------")
    print("  NoCon-coop                56%   3.06  0%")
    print("  RSS-coop                   0%   —    9%")
    print("  RiskMM-coop               33%   2.43  11%")
    print("  Hybrid-thr0.30 (no P0)    56%   3.06  41%   ← previous baseline")


if __name__ == "__main__":
    main()

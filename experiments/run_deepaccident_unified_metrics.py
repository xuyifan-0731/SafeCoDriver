"""DeepAccident eval matching the blind-spot unified eval metrics.

Reports SAME metrics as run_blindspot_unified for cross-platform comparison:
  CollRate, 2ndColl, Sev, Det(s), Det(f), Early, FA(s), FA(f), WPColl%, ModRate

Note: DeepAccident scenarios don't simulate ego control (offline data),
so 'CollRate' is from the dataset labels (52 accident, 52 normal).
'2ndColl' and 'Sev' are not measurable here.
"""
from __future__ import annotations
import sys, os, math, time, logging
import numpy as np
from pathlib import Path
from collections import defaultdict, deque

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.interface import ConstraintMode, PerceptionResult
from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import (
    simulate_codriving_waypoints, check_waypoint_collision)
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND


def eval_method(method, val_idx, loader, ego_uses_coop=True):
    """Returns dict of all metrics."""
    n_acc = 0; n_norm = 0
    n_det_acc = 0; n_fa_norm = 0
    early_warn_frames = []
    n_total_frames = 0; n_modified_frames = 0
    n_dangerous_frames = 0; n_warned_dangerous = 0
    n_safe_frames = 0; n_warned_safe = 0
    wp_coll = 0; wp_total = 0

    for si in val_idx:
        s = loader.scenarios[si]
        cf = s.get('collision_frame', -1)
        first_warn = -1; had_fa = False
        # Reset stateful
        if hasattr(method, 'recent_probs'):
            method.recent_probs.clear()
        if hasattr(method, 'recent_geom'):
            method.recent_geom.clear()

        for fi in range(len(s['frames'])):
            frame = loader.load_frame(si, fi)
            ftc = (cf - fi) if (cf > 0 and s['is_accident']) else -1
            is_dangerous = s['is_accident'] and 0 < ftc <= 30

            # Ego-only: filter invisible agents
            perception = frame.perception
            if not ego_uses_coop:
                perception = PerceptionResult(
                    timestamp=perception.timestamp,
                    ego=perception.ego,
                    agents=[a for a in perception.agents if a.is_visible],
                    lanes=perception.lanes if hasattr(perception, 'lanes') else [])

            bw = simulate_codriving_waypoints(frame)
            wm = False
            mw = bw

            if method is None:
                pass
            elif hasattr(method, 'constrain_waypoints'):
                try:
                    mw, stats = method.constrain_waypoints(bw, perception)
                    wm = (stats.get('n_collisions_detected', 0) > 0
                          or stats.get('modification_rate', 0) > 0
                          or stats.get('n_modifications', 0) > 0
                          or stats.get('n_geometric_threats', 0) > 0)
                except Exception:
                    pass
            else:
                try:
                    safe = method.constrain(perception)
                    wm = safe.mode != ConstraintMode.NORMAL
                except Exception:
                    pass

            n_total_frames += 1
            if wm: n_modified_frames += 1

            if is_dangerous:
                n_dangerous_frames += 1
                if wm: n_warned_dangerous += 1
            else:
                n_safe_frames += 1
                if wm: n_warned_safe += 1

            if wm and first_warn < 0:
                first_warn = fi
            if wm and not s['is_accident']:
                had_fa = True

            # Always check WPColl against full ground truth (including invisible agents)
            wp_coll += check_waypoint_collision(mw, frame)
        wp_total += len(s['frames']) * 10

        if s['is_accident']:
            n_acc += 1
            if first_warn >= 0:
                n_det_acc += 1
                early_warn_frames.append(len(s['frames']) - first_warn)
        else:
            n_norm += 1
            if had_fa: n_fa_norm += 1

    return {
        'CollRate': n_acc / max(n_acc + n_norm, 1),  # ground-truth collision rate
        'sec': 0,  # not measurable
        'Sev': 0.0,  # not measurable
        'Det(s)': n_det_acc / max(n_acc, 1),
        'Det(f)': n_warned_dangerous / max(n_dangerous_frames, 1),
        'Early': np.mean(early_warn_frames) if early_warn_frames else 0,
        'FA(s)': n_fa_norm / max(n_norm, 1),
        'FA(f)': n_warned_safe / max(n_safe_frames, 1),
        'WPColl': wp_coll / max(wp_total, 1),
        'Mod': n_modified_frames / max(n_total_frames, 1),
    }


def main():
    print("=" * 70)
    print("  DeepAccident All-Metrics Eval (260511)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()
    val_idx = ckpt.get('val_scenario_idx', [])

    loader = DeepAccidentLoader(split='all')
    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5,
                       base_margin_invisible=4.0)

    method_configs = [
        ("NoCon-egoonly",  lambda: None,  False),
        ("NoCon-coop",     lambda: None,  True),
        ("RSS-coop",       lambda: RSSOnly(),  True),
        ("APF-coop",       lambda: RiskPotentialField(),  True),
        ("UniE2EV2X-coop", lambda: UniE2EV2XSafety(safety_threshold=3.0), True),
        ("MAP-coop",       lambda: MAPSafety(min_clearance=0.5), True),
        ("RiskMM-coop",    lambda: RiskMMSafety(v_max=20.0), True),
    ]
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        method_configs.append(
            (f"Hybrid-thr{thr:.2f}",
             lambda thr=thr: HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
             True))
    for thr in [0.25, 0.30, 0.35, 0.40]:
        def make_and(thr=thr):
            base = HybridSafetyConstraint(detection_threshold=thr, **base_kwargs)
            return HybridWithGeometricAND(base)
        method_configs.append((f"Hybrid+AND-thr{thr:.2f}", make_and, True))

    print(f"\n  Val scenarios: {len(val_idx)}")
    print(f"  {'Method':25s} {'Det(s)':>7s} {'Det(f)':>7s} {'Early':>6s} "
          f"{'FA(s)':>6s} {'FA(f)':>6s} {'WPC%':>6s} {'Mod%':>5s}")
    print('  ' + '-' * 80)
    summary = []
    for mname, factory, uses_coop in method_configs:
        method = factory()
        res = eval_method(method, val_idx, loader, ego_uses_coop=uses_coop)
        res['name'] = mname
        summary.append(res)
        print(f"  {mname:25s} {res['Det(s)']:6.0%} {res['Det(f)']:6.0%} "
              f"{res['Early']:5.1f} {res['FA(s)']:5.0%} {res['FA(f)']:5.0%} "
              f"{res['WPColl']:5.1%} {res['Mod']:4.0%}")

    # Ranking by user priority: WPColl% > 2nd > Det(s) > FA(s) > Sev
    # Within DeepAccident: 2nd and Sev not applicable, so use:
    # WPColl% > FA(f) > Det(s) > Early
    # (FA matters more than Det because all methods get Det=100%)
    print(f"\n{'=' * 70}")
    print("  RANKING (priority: WPColl% > FA(f) > -Det(s) > FA(s) > -Early)")
    print('=' * 70)

    def rank_metric(items, key, ascending=True):
        sorted_items = sorted(items, key=key, reverse=not ascending)
        rank_map = {}
        for i, item in enumerate(sorted_items):
            rank_map[item['name']] = i
        return rank_map

    # Lower better
    rank_wpc = rank_metric(summary, lambda x: x['WPColl'])
    rank_faf = rank_metric(summary, lambda x: x['FA(f)'])
    rank_fas = rank_metric(summary, lambda x: x['FA(s)'])
    rank_det = rank_metric(summary, lambda x: -x['Det(s)'])  # higher better
    rank_early = rank_metric(summary, lambda x: -x['Early'])

    # Weighted: WPColl most important
    weights = [10, 5, 4, 2, 1]
    for s in summary:
        s['score'] = (rank_wpc[s['name']] * weights[0] +
                      rank_faf[s['name']] * weights[1] +
                      rank_det[s['name']] * weights[2] +
                      rank_fas[s['name']] * weights[3] +
                      rank_early[s['name']] * weights[4])
    summary.sort(key=lambda x: x['score'])
    print(f"  {'Rank':>4s}  {'Method':25s} {'Score':>5s} {'WPC%':>6s} "
          f"{'FA(f)':>6s} {'Det(s)':>7s} {'Early':>6s}")
    print('  ' + '-' * 65)
    for i, s in enumerate(summary):
        print(f"  {i+1:>4d}  {s['name']:25s} {s['score']:4d} "
              f"{s['WPColl']:5.1%} {s['FA(f)']:5.1%} {s['Det(s)']:6.0%} {s['Early']:5.1f}")


if __name__ == "__main__":
    main()

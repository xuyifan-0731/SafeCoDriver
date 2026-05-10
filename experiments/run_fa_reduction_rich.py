"""Comprehensive 260510 experiment: SUMO forced-conflict + FA reduction + per-frame metrics.

Phase 1: Forced-conflict SUMO (single-lane, attacker forces collision)
Phase 2: FA reduction with PER-FRAME FA metric (shows AND/multi-frame benefit)
Phase 3: TTC-windowed FA (only count alerts when no real danger nearby)
"""
from __future__ import annotations
import sys, os, math, time, random, logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

sys.path.insert(0, str(Path(__file__).parent.parent))
from coop_safety.interface import (PerceptionResult, VehicleState, Agent,
                                    AgentType, ConstraintMode)
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety
from experiments.run_corrected_eval import build_perception, get_veh
from experiments.run_forced_conflict_and_fa import (
    generate_forced_conflict, run_sumo_forced,
    HybridWithGeometricAND, HybridWithMultiframe, HybridWithBothAND3frame)

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
NEW_V1_THRESHOLD = 0.25


# =================================================================
# Phase 2 & 3: FA reduction with per-frame metrics
# =================================================================

@dataclass
class FAResult:
    name: str = ""
    n_acc_scen: int = 0
    n_norm_scen: int = 0
    n_detected_scen: int = 0
    n_fa_scen: int = 0  # Scenarios with any FA
    early_warning_frames: list = field(default_factory=list)
    # Per-frame metrics
    n_fired_frames_normal: int = 0  # Per-frame FA in normal scenarios
    n_total_frames_normal: int = 0
    n_fired_frames_acc: int = 0  # Per-frame TP in accident scenarios
    n_total_frames_acc: int = 0
    n_dangerous_frames_acc: int = 0  # Frames within collision window
    n_fired_dangerous: int = 0       # Fired during dangerous window
    n_fired_safe_in_acc: int = 0     # Fired in non-dangerous frames of acc scenarios
    n_safe_frames_acc: int = 0
    # Waypoints
    waypoint_collisions: int = 0
    total_waypoint_checks: int = 0


def eval_fa_methods(method, val_idx, loader):
    """Evaluate one method with rich FA metrics."""
    from experiments.run_deepaccident_unified import (
        simulate_codriving_waypoints, check_waypoint_collision)

    r = FAResult(name=getattr(method, 'name', '?'))

    for si in val_idx:
        s = loader.scenarios[si]
        # Reset state
        if hasattr(method, 'recent_probs'):
            method.recent_probs.clear()
        if hasattr(method, 'recent_geom'):
            method.recent_geom.clear()

        first_warning = -1
        had_fa = False
        cf = s.get('collision_frame', -1)

        for fi in range(len(s['frames'])):
            frame = loader.load_frame(si, fi)
            bw = simulate_codriving_waypoints(frame)
            mw, stats = method.constrain_waypoints(bw, frame.perception)
            fired = stats.get('n_collisions_detected', 0) > 0

            ftc = (cf - fi) if (cf > 0 and s['is_accident']) else -1
            is_dangerous_frame = s['is_accident'] and 0 < ftc <= 30

            if s['is_accident']:
                r.n_total_frames_acc += 1
                if is_dangerous_frame:
                    r.n_dangerous_frames_acc += 1
                else:
                    r.n_safe_frames_acc += 1
                if fired:
                    r.n_fired_frames_acc += 1
                    if is_dangerous_frame:
                        r.n_fired_dangerous += 1
                    else:
                        r.n_fired_safe_in_acc += 1
                    if first_warning < 0:
                        first_warning = fi
            else:
                r.n_total_frames_normal += 1
                if fired:
                    r.n_fired_frames_normal += 1
                    had_fa = True
                    if first_warning < 0:
                        first_warning = fi

            r.waypoint_collisions += check_waypoint_collision(mw, frame)
        r.total_waypoint_checks += len(s['frames']) * 10

        if s['is_accident']:
            r.n_acc_scen += 1
            if first_warning >= 0:
                r.n_detected_scen += 1
                r.early_warning_frames.append(len(s['frames']) - first_warning)
        else:
            r.n_norm_scen += 1
            if had_fa:
                r.n_fa_scen += 1

    return r


def print_fa_table(results, label):
    print(f"\n  --- {label} ---")
    print(f"  {'Method':28s} {'Det':>5s} {'Early':>6s} "
          f"{'FA(scen)':>9s} {'FA(frame)':>10s} {'PT-fire%':>9s} {'Frame-Pre':>10s} {'WPC%':>5s}")
    print("  " + "-" * 95)
    for r in results:
        det = r.n_detected_scen / max(r.n_acc_scen, 1)
        early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa_scen = r.n_fa_scen / max(r.n_norm_scen, 1)
        fa_frame = r.n_fired_frames_normal / max(r.n_total_frames_normal, 1)
        # Pre-trigger fire rate: fires while in dangerous window in accident scenarios
        pretrigger = r.n_fired_dangerous / max(r.n_dangerous_frames_acc, 1)
        # Frame-level precision: TP / (TP + FP), where TP = fires in dangerous, FP = fires not in dangerous
        # Across all scenarios
        total_fires = r.n_fired_frames_normal + r.n_fired_frames_acc
        true_positive = r.n_fired_dangerous
        false_positive = total_fires - true_positive
        frame_pre = true_positive / max(true_positive + false_positive, 1)
        wpc = r.waypoint_collisions / max(r.total_waypoint_checks, 1)
        print(f"  {r.name:28s} {det:4.0%} {early:5.1f} "
              f"{fa_scen:8.1%} {fa_frame:9.1%} {pretrigger:8.1%} {frame_pre:9.1%} {wpc:4.1%}")


def main():
    print("=" * 70)
    print("  Phase 2: FA Reduction with Rich Per-Frame Metrics (260510)")
    print("=" * 70)
    print("  Metrics:")
    print("    Det        = scenario-level detection rate (in accident set)")
    print("    Early      = average early warning frames (det - first_warning)")
    print("    FA(scen)   = scenarios with at least one false alarm / total normal")
    print("    FA(frame)  = total FA frames / total normal frames")
    print("    PT-fire%   = pre-trigger fire rate (fired during dangerous window)")
    print("    Frame-Pre  = frame-level precision: TP / (TP + FP)")
    print("    WPC%       = waypoint collision rate")

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"\n  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    val_idx = ckpt.get('val_scenario_idx', [])

    from experiments.deepaccident_loader import DeepAccidentLoader
    loader = DeepAccidentLoader(split='all')

    # ============ A) Threshold sweep ============
    threshold_sweep = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    res_thr = []
    for thr in threshold_sweep:
        method = HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=thr)
        method.name = f"Hybrid-thr{thr:.2f}"
        r = eval_fa_methods(method, val_idx, loader)
        res_thr.append(r)
    print_fa_table(res_thr, "A) Threshold sweep on Ours-Hybrid")

    # ============ B) Variants at threshold=0.40 (best from A) ============
    base = HybridSafetyConstraint(
        detector_model=v1, base_margin_visible=2.5,
        base_margin_invisible=4.0, detection_threshold=0.40)
    base.name = "Hybrid (thr=0.40)"
    variants = [
        base,
        HybridWithGeometricAND(base),
        HybridWithMultiframe(base, n_consensus=3),
        HybridWithMultiframe(base, n_consensus=5),
        HybridWithBothAND3frame(base, n_consensus=3),
    ]
    res_var = [eval_fa_methods(v, val_idx, loader) for v in variants]
    print_fa_table(res_var, "B) Variants at threshold=0.40")

    # ============ C) Combined: threshold sweep × AND ============
    print("\n  --- C) AND-fusion at different thresholds ---")
    res_and = []
    for thr in [0.25, 0.30, 0.35, 0.40]:
        b = HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=thr)
        b.name = f"Hybrid+AND-thr{thr:.2f}"
        ag = HybridWithGeometricAND(b)
        ag.name = b.name
        r = eval_fa_methods(ag, val_idx, loader)
        res_and.append(r)
    print_fa_table(res_and, "")


if __name__ == "__main__":
    main()

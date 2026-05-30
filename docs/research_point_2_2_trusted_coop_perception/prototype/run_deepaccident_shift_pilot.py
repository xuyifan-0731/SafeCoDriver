#!/usr/bin/env python3
"""Run a small DeepAccident shift pilot for TrustCalib.

This script evaluates whether object-level translation calibration can recover
shifted cooperative blind-spot information before it reaches SafeCoDriver.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SAFECO_ROOT = Path("/raid/xuyifan/jiqiuyu")
sys.path.insert(0, str(SAFECO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
torch.set_num_threads(1)

from coop_safety.interface import PerceptionResult
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.trust_calib import (
    calibrate_shifted_coop,
    filter_visible,
    merge_ego_visible_with_coop_invisible,
    oracle_correct_message,
    shifted_message,
)


def load_hybrid(use_v1: bool = True) -> HybridSafetyConstraint:
    if not use_v1:
        return HybridSafetyConstraint(
            detector_model=None,
            detection_threshold=0.30,
            base_margin_visible=2.5,
            base_margin_invisible=4.0,
        )
    ckpt_path = SAFECO_ROOT / "models" / "collision_net_best.pt"
    model = CollisionPredictionNetwork()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return HybridSafetyConstraint(
        detector_model=model,
        detection_threshold=0.30,
        base_margin_visible=2.5,
        base_margin_invisible=4.0,
    )


def eval_perception(method: HybridSafetyConstraint, waypoints: np.ndarray, perception: PerceptionResult):
    modified, stats = method.constrain_waypoints(waypoints, perception)
    warned = (
        stats.get("n_collisions_detected", 0) > 0
        or stats.get("modification_rate", 0) > 0
        or stats.get("n_geometric_threats", 0) > 0
    )
    return modified, stats, warned


def build_method_perceptions(frame, shift_x: float, shift_y: float):
    ego_only = filter_visible(frame.perception)
    coop_shifted = shifted_message(frame.perception, shift_x, shift_y)
    raw_coop = merge_ego_visible_with_coop_invisible(frame.perception, coop_shifted)
    hard_filter = ego_only
    trust_calib, report = calibrate_shifted_coop(frame.perception, coop_shifted)
    oracle_msg = oracle_correct_message(coop_shifted, shift_x, shift_y)
    oracle = merge_ego_visible_with_coop_invisible(frame.perception, oracle_msg)
    return {
        "EgoOnly": (ego_only, None),
        "RawShiftedCoop": (raw_coop, None),
        "HardFilter": (hard_filter, None),
        "TrustCalib": (trust_calib, report),
        "OracleCalib": (oracle, None),
    }


def update_metrics(metrics, method_name, frame, modified_wp, warned, stats, report):
    m = metrics[method_name]
    m["frames"] += 1
    if warned:
        m["warned_frames"] += 1
    m["wp_collisions"] += check_waypoint_collision(modified_wp, frame)
    m["wp_total"] += len(modified_wp)
    m["geom_threats"] += stats.get("n_geometric_threats", 0)
    m["collision_prob_sum"] += stats.get("collision_prob", 0.0)
    m["mod_sum"] += stats.get("modification_rate", 0.0)

    if report is not None:
        m["reports"] += 1
        m["correct_actions"] += 1 if report.action == "correct" else 0
        m["downweight_actions"] += 1 if report.action == "downweight" else 0
        m["quarantine_actions"] += 1 if report.action == "quarantine" else 0
        m["offset_dx_sum"] += report.offset.dx
        m["offset_dy_sum"] += report.offset.dy
        m["residual_before_sum"] += report.offset.residual_before
        m["residual_after_sum"] += report.offset.residual_after
        m["correctable_sum"] += report.offset.correctable_score
        m["match_count_sum"] += report.offset.match_count


def make_metrics(method_names):
    return {
        name: {
            "frames": 0,
            "warned_frames": 0,
            "wp_collisions": 0,
            "wp_total": 0,
            "geom_threats": 0,
            "collision_prob_sum": 0.0,
            "mod_sum": 0.0,
            "reports": 0,
            "correct_actions": 0,
            "downweight_actions": 0,
            "quarantine_actions": 0,
            "offset_dx_sum": 0.0,
            "offset_dy_sum": 0.0,
            "residual_before_sum": 0.0,
            "residual_after_sum": 0.0,
            "correctable_sum": 0.0,
            "match_count_sum": 0.0,
        }
        for name in method_names
    }


def summarize(metrics):
    rows = []
    for name, m in metrics.items():
        frames = max(m["frames"], 1)
        reports = max(m["reports"], 1)
        rows.append(
            {
                "method": name,
                "frames": m["frames"],
                "warn_rate": m["warned_frames"] / frames,
                "WPC": m["wp_collisions"] / max(m["wp_total"], 1),
                "avg_mod": m["mod_sum"] / frames,
                "avg_collision_prob": m["collision_prob_sum"] / frames,
                "avg_geom_threats": m["geom_threats"] / frames,
                "reports": m["reports"],
                "correct_rate": m["correct_actions"] / reports if m["reports"] else 0.0,
                "downweight_rate": m["downweight_actions"] / reports if m["reports"] else 0.0,
                "quarantine_rate": m["quarantine_actions"] / reports if m["reports"] else 0.0,
                "avg_est_dx": m["offset_dx_sum"] / reports if m["reports"] else 0.0,
                "avg_est_dy": m["offset_dy_sum"] / reports if m["reports"] else 0.0,
                "avg_residual_before": m["residual_before_sum"] / reports if m["reports"] else 0.0,
                "avg_residual_after": m["residual_after_sum"] / reports if m["reports"] else 0.0,
                "avg_correctable": m["correctable_sum"] / reports if m["reports"] else 0.0,
                "avg_match_count": m["match_count_sum"] / reports if m["reports"] else 0.0,
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shift-x", type=float, default=2.0)
    parser.add_argument("--shift-y", type=float, default=1.0)
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--max-frames-per-scenario", type=int, default=0)
    parser.add_argument("--disable-v1", action="store_true", help="Skip V1 detector for fast geometry/WPC pilot.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/deepaccident_shift_pilot"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hybrid = load_hybrid(use_v1=not args.disable_v1)
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=False)

    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]

    method_names = ["EgoOnly", "RawShiftedCoop", "HardFilter", "TrustCalib", "OracleCalib"]
    metrics = make_metrics(method_names)

    for si in val_idx:
        scenario = loader.scenarios[si]
        n_frames = len(scenario["frames"])
        if args.max_frames_per_scenario > 0:
            n_frames = min(n_frames, args.max_frames_per_scenario)
        for fi in range(n_frames):
            frame = loader.load_frame(si, fi)
            waypoints = simulate_codriving_waypoints(frame)
            per_method = build_method_perceptions(frame, args.shift_x, args.shift_y)
            for name, (perception, report) in per_method.items():
                modified, stats, warned = eval_perception(hybrid, waypoints, perception)
                update_metrics(metrics, name, frame, modified, warned, stats, report)

    rows = summarize(metrics)
    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "shift_x": args.shift_x,
        "shift_y": args.shift_y,
        "max_scenarios": args.max_scenarios,
        "max_frames_per_scenario": args.max_frames_per_scenario,
        "disable_v1": bool(args.disable_v1),
        "val_scenarios_used": len(val_idx),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Shift: dx={args.shift_x:.2f}, dy={args.shift_y:.2f}")
    print(f"Scenarios: {len(val_idx)}")
    print(f"{'method':18s} {'WPC%':>8s} {'warn%':>8s} {'mod%':>8s} {'p_coll':>8s} {'est_dx':>8s} {'est_dy':>8s} {'corr':>8s}")
    for r in rows:
        print(
            f"{r['method']:18s} {100*r['WPC']:7.2f}% {100*r['warn_rate']:7.2f}% "
            f"{100*r['avg_mod']:7.2f}% {r['avg_collision_prob']:8.3f} "
            f"{r['avg_est_dx']:8.3f} {r['avg_est_dy']:8.3f} {r['avg_correctable']:8.3f}"
        )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

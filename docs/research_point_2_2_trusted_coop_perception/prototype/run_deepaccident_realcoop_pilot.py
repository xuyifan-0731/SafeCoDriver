#!/usr/bin/env python3
"""Evaluate aligned real DeepAccident other_vehicle cooperative labels."""
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

from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.real_coop import filter_sender_visible, load_aligned_other_vehicle_message, merge_ego_visible_with_real_other
from prototype.run_deepaccident_mixed_pilot import eval_method, load_hybrid, object_impact_guard_perception
from prototype.trust_calib import calibrate_shifted_coop, filter_visible, merge_ego_visible_with_coop_invisible


class NoPeerEvidence:
    """Conservative evidence model for one real cooperative sender."""

    def is_supported(self, agent) -> bool:
        return bool(agent.is_visible)


def geometry_match(agent, candidates, max_dist: float = 2.5, max_size_diff: float = 3.0):
    best = None
    best_cost = float("inf")
    for candidate in candidates:
        if candidate.agent_type != agent.agent_type:
            continue
        dist = float(np.hypot(candidate.state.x - agent.state.x, candidate.state.y - agent.state.y))
        if dist > max_dist:
            continue
        size = abs(candidate.state.length - agent.state.length) + abs(candidate.state.width - agent.state.width)
        if size > max_size_diff:
            continue
        cost = dist + 0.5 * size
        if cost < best_cost:
            best = candidate
            best_cost = cost
    return best


def coop_coverage_stats(clean_coop, real_coop):
    clean_blind = [agent for agent in clean_coop.agents if agent.source == "coop"]
    real_blind = [agent for agent in real_coop.agents if agent.source == "coop"]
    matched = 0
    for agent in clean_blind:
        if geometry_match(agent, real_blind) is not None:
            matched += 1
    extra = 0
    for agent in real_blind:
        if geometry_match(agent, clean_blind) is None:
            extra += 1
    return len(clean_blind), len(real_blind), matched, extra


def init_metrics(names):
    return {
        name: {
            "frames": 0,
            "frames_with_real": 0,
            "warned": 0,
            "wp_coll": 0,
            "wp_total": 0,
            "mod_sum": 0.0,
            "geom_sum": 0,
            "p_sum": 0.0,
            "reports": 0,
            "accept": 0,
            "correct": 0,
            "downweight": 0,
            "quarantine": 0,
            "object_quarantine": 0,
            "residual_after_sum": 0.0,
            "match_count_sum": 0.0,
            "obj_removed_sum": 0,
            "clean_blind_sum": 0,
            "real_blind_sum": 0,
            "matched_blind_sum": 0,
            "extra_real_sum": 0,
        }
        for name in names
    }


def update(metrics, name, frame, modified, stats, warned, report=None, obj_removed=0, coverage=None, has_real=True):
    m = metrics[name]
    m["frames"] += 1
    m["frames_with_real"] += int(has_real)
    m["warned"] += int(warned)
    m["wp_coll"] += check_waypoint_collision(modified, frame)
    m["wp_total"] += len(modified)
    m["mod_sum"] += stats.get("modification_rate", 0.0)
    m["geom_sum"] += stats.get("n_geometric_threats", 0)
    m["p_sum"] += stats.get("collision_prob", 0.0)
    m["obj_removed_sum"] += obj_removed
    if report is not None:
        m["reports"] += 1
        if report.action in m:
            m[report.action] += 1
        m["residual_after_sum"] += report.offset.residual_after
        m["match_count_sum"] += report.offset.match_count
    if coverage is not None:
        clean_blind, real_blind, matched_blind, extra_real = coverage
        m["clean_blind_sum"] += clean_blind
        m["real_blind_sum"] += real_blind
        m["matched_blind_sum"] += matched_blind
        m["extra_real_sum"] += extra_real


def rows_from_metrics(metrics):
    rows = []
    for name, m in metrics.items():
        frames = max(m["frames"], 1)
        reports = max(m["reports"], 1)
        rows.append(
            {
                "method": name,
                "frames": m["frames"],
                "frames_with_real": m["frames_with_real"],
                "real_coverage": m["frames_with_real"] / frames,
                "wp_coll": m["wp_coll"],
                "wp_total": m["wp_total"],
                "WPC": m["wp_coll"] / max(m["wp_total"], 1),
                "warn_rate": m["warned"] / frames,
                "avg_mod": m["mod_sum"] / frames,
                "avg_geom": m["geom_sum"] / frames,
                "avg_p_coll": m["p_sum"] / frames,
                "reports": m["reports"],
                "accept_rate": m["accept"] / reports if m["reports"] else 0.0,
                "correct_rate": m["correct"] / reports if m["reports"] else 0.0,
                "downweight_rate": m["downweight"] / reports if m["reports"] else 0.0,
                "quarantine_rate": m["quarantine"] / reports if m["reports"] else 0.0,
                "object_quarantine_rate": m["object_quarantine"] / reports if m["reports"] else 0.0,
                "avg_residual_after": m["residual_after_sum"] / reports if m["reports"] else 0.0,
                "avg_match_count": m["match_count_sum"] / reports if m["reports"] else 0.0,
                "avg_obj_removed": m["obj_removed_sum"] / frames,
                "avg_clean_blind": m["clean_blind_sum"] / frames,
                "avg_real_blind": m["real_blind_sum"] / frames,
                "blind_recall": m["matched_blind_sum"] / max(m["clean_blind_sum"], 1),
                "extra_real_per_frame": m["extra_real_sum"] / frames,
            }
        )
    return rows


def run(args):
    hybrid = load_hybrid(use_v1=not args.disable_v1)
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=True)
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    available_val_scenarios = len(val_idx)
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]

    names = ["EgoOnly", "CleanCoop", "RealOtherRaw", "RealOtherTrustCalib", "RealOtherObjectGuard"]
    metrics = init_metrics(names)

    for si in val_idx:
        scenario = loader.scenarios[si]
        n_frames = len(scenario["frames"])
        if args.max_frames_per_scenario > 0:
            n_frames = min(n_frames, args.max_frames_per_scenario)
        for fi in range(n_frames):
            frame = loader.load_frame(si, fi)
            waypoints = simulate_codriving_waypoints(frame)
            ego_only = filter_visible(frame.perception)
            clean_coop = merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)
            aligned_other = load_aligned_other_vehicle_message(loader, si, fi)
            if aligned_other is None:
                real_raw = ego_only
                real_trust = ego_only
                real_report = None
                real_guard = ego_only
                guard_report = None
                obj_removed = 0
                coverage = (0, 0, 0, 0)
                has_real = False
            else:
                aligned_other = filter_sender_visible(aligned_other)
                real_raw = merge_ego_visible_with_real_other(frame.perception, aligned_other)
                real_trust, real_report = calibrate_shifted_coop(
                    frame.perception,
                    aligned_other,
                    clean_residual_thr=args.clean_residual_thr,
                )
                real_guard, guard_report, obj_removed = object_impact_guard_perception(
                    ego_only,
                    real_trust,
                    real_report,
                    hybrid,
                    waypoints,
                    evidence_tracker=NoPeerEvidence(),
                    geom_delta_thr=args.real_guard_geom_delta_thr,
                    mod_delta_thr=args.real_guard_mod_delta_thr,
                )
                coverage = coop_coverage_stats(clean_coop, real_raw)
                has_real = True

            per_method = {
                "EgoOnly": (ego_only, None, 0),
                "CleanCoop": (clean_coop, None, 0),
                "RealOtherRaw": (real_raw, None, 0),
                "RealOtherTrustCalib": (real_trust, real_report, 0),
                "RealOtherObjectGuard": (real_guard, guard_report, obj_removed),
            }
            for name, (perception, report, removed) in per_method.items():
                modified, stats, warned = eval_method(hybrid, waypoints, perception)
                update(metrics, name, frame, modified, stats, warned, report, removed, coverage, has_real)

    rows = rows_from_metrics(metrics)
    for row in rows:
        row["actual_scenarios"] = len(val_idx)
        row["available_val_scenarios"] = available_val_scenarios
        row["requested_max_scenarios"] = args.max_scenarios
        row["max_frames_per_scenario"] = args.max_frames_per_scenario
        row["disable_v1"] = bool(args.disable_v1)
    return rows, available_val_scenarios, val_idx


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-scenarios", type=int, default=22)
    parser.add_argument("--max-frames-per-scenario", type=int, default=20)
    parser.add_argument("--clean-residual-thr", type=float, default=0.50)
    parser.add_argument("--real-guard-geom-delta-thr", type=float, default=1.0)
    parser.add_argument("--real-guard-mod-delta-thr", type=float, default=0.10)
    parser.add_argument("--disable-v1", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/realcoop_pilot"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, available_val_scenarios, val_idx = run(args)

    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = vars(args).copy()
    meta["available_val_scenarios"] = available_val_scenarios
    meta["val_scenarios_used"] = len(val_idx)
    meta["val_scenario_idx_used"] = val_idx
    meta["out_dir"] = str(args.out_dir)
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"{'method':22s} {'WPC%':>7s} {'warn%':>7s} {'blindR':>7s} {'real/fr':>7s} {'extra/fr':>8s} {'accept':>7s} {'resid':>8s}")
    for row in rows:
        print(
            f"{row['method']:22s} {100*row['WPC']:6.3f}% {100*row['warn_rate']:6.2f}% "
            f"{100*row['blind_recall']:6.2f}% {row['avg_real_blind']:7.2f} "
            f"{row['extra_real_per_frame']:8.2f} {100*row['accept_rate']:6.2f}% "
            f"{row['avg_residual_after']:8.4f}"
        )
    print(f"Wrote {args.out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

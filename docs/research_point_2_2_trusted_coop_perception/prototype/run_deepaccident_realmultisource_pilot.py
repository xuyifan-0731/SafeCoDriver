#!/usr/bin/env python3
"""Evaluate aligned real multi-source DeepAccident cooperative labels."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAFECO_ROOT = Path("/raid/xuyifan/jiqiuyu")
sys.path.insert(0, str(SAFECO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

torch.set_num_threads(1)

from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.real_coop import filter_sender_visible, load_aligned_source_message, merge_ego_visible_with_real_other
from prototype.run_deepaccident_mixed_pilot import eval_method, load_hybrid, object_impact_guard_perception
from prototype.run_deepaccident_multipeer_pilot import (
    MultiPeerEvidenceSupport,
    apply_box_margin_guard,
    apply_missing_object_recovery,
    build_cluster_records,
)
from prototype.trust_calib import calibrate_shifted_coop, filter_visible, merge_ego_visible_with_coop_invisible


def init_metrics(names):
    return {
        name: {
            "frames": 0,
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
            "obj_removed_sum": 0,
            "missing_recovered_sum": 0,
            "missing_candidate_sum": 0,
            "missing_gt_sum": 0,
            "missing_tp_sum": 0,
            "missing_fp_sum": 0,
            "source_available_sum": 0,
        }
        for name in names
    }


def update(metrics, name, frame, modified, stats, warned, report=None, obj_removed=0, missing=None, source_available=0):
    m = metrics[name]
    m["frames"] += 1
    m["warned"] += int(warned)
    m["wp_coll"] += check_waypoint_collision(modified, frame)
    m["wp_total"] += len(modified)
    m["mod_sum"] += stats.get("modification_rate", 0.0)
    m["geom_sum"] += stats.get("n_geometric_threats", 0)
    m["p_sum"] += stats.get("collision_prob", 0.0)
    m["obj_removed_sum"] += obj_removed
    m["source_available_sum"] += source_available
    if report is not None:
        m["reports"] += 1
        if report.action in m:
            m[report.action] += 1
    if missing is not None:
        recovered, candidates, gt, tp, fp = missing
        m["missing_recovered_sum"] += recovered
        m["missing_candidate_sum"] += candidates
        m["missing_gt_sum"] += gt
        m["missing_tp_sum"] += tp
        m["missing_fp_sum"] += fp


def rows_from_metrics(metrics):
    rows = []
    for name, m in metrics.items():
        frames = max(m["frames"], 1)
        reports = max(m["reports"], 1)
        rows.append(
            {
                "method": name,
                "frames": m["frames"],
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
                "avg_obj_removed": m["obj_removed_sum"] / frames,
                "avg_missing_recovered": m["missing_recovered_sum"] / frames,
                "avg_missing_candidates": m["missing_candidate_sum"] / frames,
                "avg_missing_gt": m["missing_gt_sum"] / frames,
                "missing_precision": m["missing_tp_sum"] / max(m["missing_tp_sum"] + m["missing_fp_sum"], 1),
                "missing_recall": m["missing_tp_sum"] / max(m["missing_gt_sum"], 1),
                "avg_sources_available": m["source_available_sum"] / frames,
            }
        )
    return rows


def make_missing_args(args):
    return SimpleNamespace(
        enable_missing_recovery=True,
        missing_recovery_primary_actions=args.missing_recovery_primary_actions,
        missing_min_peer_support=args.missing_min_peer_support,
        missing_min_trust_support=args.missing_min_trust_support,
        missing_match_dist=args.missing_match_dist,
        missing_cluster_dist=args.missing_cluster_dist,
        missing_max_size_diff=args.missing_max_size_diff,
        missing_recover_high_impact_only=False,
        missing_impact_geom_thr=1.0,
        missing_impact_mod_thr=0.1,
        path_box_ego_radius=args.path_box_ego_radius,
        enable_box_margin_guard=args.enable_box_margin_guard,
        box_margin_guard_thr=args.box_margin_guard_thr,
    )


def run(args):
    hybrid = load_hybrid(use_v1=not args.disable_v1)
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=True)
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    available_val_scenarios = len(val_idx)
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]

    source_roles = [role.strip() for role in args.source_roles.split(",") if role.strip()]
    if args.primary_role not in source_roles:
        source_roles.insert(0, args.primary_role)
    support_roles = [role for role in source_roles if role != args.primary_role]

    names = ["EgoOnly", "CleanCoop", "RealPrimaryRaw", "RealPrimaryTrustCalib", "RealMultiEvidenceGuard"]
    metrics = init_metrics(names)
    missing_args = make_missing_args(args)
    cluster_records = []
    frame_records = []

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

            source_messages = {}
            for role in source_roles:
                msg = load_aligned_source_message(loader, si, fi, source_role=role)
                if msg is not None:
                    source_messages[role] = filter_sender_visible(msg)

            primary_msg = source_messages.get(args.primary_role)
            source_available = len(source_messages)
            if primary_msg is None:
                primary_raw = ego_only
                primary_trust = ego_only
                primary_report = None
                guarded = ego_only
                guarded_report = None
                obj_removed = 0
                missing_tuple = (0, 0, 0, 0, 0)
            else:
                primary_raw = merge_ego_visible_with_real_other(frame.perception, primary_msg)
                primary_trust, primary_report = calibrate_shifted_coop(
                    frame.perception,
                    primary_msg,
                    clean_residual_thr=args.clean_residual_thr,
                )
                support_trust = []
                support_reports = []
                support_ids = []
                for role in support_roles:
                    support_msg = source_messages.get(role)
                    if support_msg is None:
                        continue
                    perception, report = calibrate_shifted_coop(
                        frame.perception,
                        support_msg,
                        clean_residual_thr=args.clean_residual_thr,
                    )
                    support_trust.append(perception)
                    support_reports.append(report)
                    support_ids.append(role)

                evidence = MultiPeerEvidenceSupport(
                    support_trust,
                    peer_reports=support_reports,
                    sender_ids=support_ids,
                    sender_trusts=[1.0] * len(support_trust),
                    min_support=args.min_peer_support,
                    min_trust_support=args.min_trust_support,
                    max_dist=args.peer_match_dist,
                )
                guarded, guarded_report, obj_removed = object_impact_guard_perception(
                    ego_only,
                    primary_trust,
                    primary_report,
                    hybrid,
                    waypoints,
                    evidence_tracker=evidence,
                    geom_delta_thr=args.guard_geom_delta_thr,
                    mod_delta_thr=args.guard_mod_delta_thr,
                )
                guarded_report = copy.deepcopy(guarded_report)
                guarded, n_box_removed = apply_box_margin_guard(guarded, evidence, missing_args, waypoints)
                if n_box_removed:
                    obj_removed += n_box_removed
                    guarded_report.action = "object_quarantine"
                    guarded_report.n_output_agents = len(guarded.agents)

                (
                    guarded,
                    n_missing_recovered,
                    n_missing_candidates,
                    n_missing_gt,
                    n_missing_tp,
                    n_missing_fp,
                    missing_records,
                ) = apply_missing_object_recovery(
                    "real_multisource",
                    si,
                    fi,
                    guarded,
                    primary_trust,
                    primary_report,
                    clean_coop,
                    support_trust,
                    support_ids,
                    [1.0] * len(support_trust),
                    missing_args,
                    ego_only,
                    hybrid,
                    waypoints,
                )
                missing_tuple = (n_missing_recovered, n_missing_candidates, n_missing_gt, n_missing_tp, n_missing_fp)
                if args.write_diagnostics:
                    records = build_cluster_records(
                        "real_multisource",
                        si,
                        fi,
                        args.primary_role,
                        primary_trust,
                        guarded,
                        evidence,
                        {},
                        {},
                        ego_only,
                        hybrid,
                        waypoints,
                        missing_args,
                    )
                    for record in records:
                        record["primary_role"] = args.primary_role
                        record["primary_action"] = primary_report.action
                        record["source_available"] = source_available
                    for record in missing_records:
                        record["primary_role"] = args.primary_role
                        record["primary_action"] = primary_report.action
                        record["source_available"] = source_available
                    cluster_records.extend(records)
                    cluster_records.extend(missing_records)

            per_method = {
                "EgoOnly": (ego_only, None, 0, None),
                "CleanCoop": (clean_coop, None, 0, None),
                "RealPrimaryRaw": (primary_raw, None, 0, None),
                "RealPrimaryTrustCalib": (primary_trust, primary_report, 0, None),
                "RealMultiEvidenceGuard": (guarded, guarded_report, obj_removed, missing_tuple),
            }
            frame_record = {
                "scenario_index": si,
                "frame_index": fi,
                "source_available": source_available,
                "primary_action": primary_report.action if primary_report is not None else "missing_primary",
                "guarded_action": guarded_report.action if guarded_report is not None else "none",
                "obj_removed": obj_removed,
                "missing_recovered": missing_tuple[0] if missing_tuple is not None else 0,
                "missing_candidates": missing_tuple[1] if missing_tuple is not None else 0,
                "missing_gt": missing_tuple[2] if missing_tuple is not None else 0,
                "missing_tp": missing_tuple[3] if missing_tuple is not None else 0,
                "missing_fp": missing_tuple[4] if missing_tuple is not None else 0,
            }
            for name, (perception, report, removed, missing_tuple_value) in per_method.items():
                modified, stats, warned = eval_method(hybrid, waypoints, perception)
                if args.write_diagnostics:
                    frame_record[f"{name}_wp_coll"] = check_waypoint_collision(modified, frame)
                    frame_record[f"{name}_warned"] = int(warned)
                    frame_record[f"{name}_geom"] = stats.get("n_geometric_threats", 0)
                    frame_record[f"{name}_mod"] = stats.get("modification_rate", 0.0)
                    frame_record[f"{name}_agents"] = len(perception.agents)
                update(
                    metrics,
                    name,
                    frame,
                    modified,
                    stats,
                    warned,
                    report=report,
                    obj_removed=removed,
                    missing=missing_tuple_value,
                    source_available=source_available,
                )
            if args.write_diagnostics:
                frame_record["guard_minus_ego_wp_coll"] = (
                    frame_record["RealMultiEvidenceGuard_wp_coll"] - frame_record["EgoOnly_wp_coll"]
                )
                frame_record["guard_minus_clean_wp_coll"] = (
                    frame_record["RealMultiEvidenceGuard_wp_coll"] - frame_record["CleanCoop_wp_coll"]
                )
                frame_records.append(frame_record)

    rows = rows_from_metrics(metrics)
    for row in rows:
        row["primary_role"] = args.primary_role
        row["source_roles"] = ",".join(source_roles)
        row["actual_scenarios"] = len(val_idx)
        row["available_val_scenarios"] = available_val_scenarios
        row["requested_max_scenarios"] = args.max_scenarios
        row["max_frames_per_scenario"] = args.max_frames_per_scenario
        row["disable_v1"] = bool(args.disable_v1)
    return rows, val_idx, available_val_scenarios, source_roles, cluster_records, frame_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roles", default="other_vehicle,infrastructure,ego_vehicle_behind,other_vehicle_behind")
    parser.add_argument("--primary-role", default="other_vehicle")
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--max-frames-per-scenario", type=int, default=20)
    parser.add_argument("--clean-residual-thr", type=float, default=0.50)
    parser.add_argument("--min-peer-support", type=int, default=1)
    parser.add_argument("--min-trust-support", type=float, default=1.0)
    parser.add_argument("--peer-match-dist", type=float, default=2.5)
    parser.add_argument("--guard-geom-delta-thr", type=float, default=1.0)
    parser.add_argument("--guard-mod-delta-thr", type=float, default=0.10)
    parser.add_argument("--enable-box-margin-guard", action="store_true")
    parser.add_argument("--box-margin-guard-thr", type=float, default=0.0)
    parser.add_argument("--path-box-ego-radius", type=float, default=1.0)
    parser.add_argument("--missing-recovery-primary-actions", default="accept,correct,downweight")
    parser.add_argument("--missing-min-peer-support", type=int, default=1)
    parser.add_argument("--missing-min-trust-support", type=float, default=1.0)
    parser.add_argument("--missing-match-dist", type=float, default=2.5)
    parser.add_argument("--missing-cluster-dist", type=float, default=2.5)
    parser.add_argument("--missing-max-size-diff", type=float, default=3.0)
    parser.add_argument("--disable-v1", action="store_true")
    parser.add_argument("--write-diagnostics", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/realmultisource_pilot"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows, val_idx, available_val_scenarios, source_roles, cluster_records, frame_records = run(args)

    with (args.out_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta = vars(args).copy()
    meta["source_roles"] = source_roles
    meta["available_val_scenarios"] = available_val_scenarios
    meta["val_scenarios_used"] = len(val_idx)
    meta["val_scenario_idx_used"] = val_idx
    meta["out_dir"] = str(args.out_dir)
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if frame_records:
        frame_fieldnames = []
        for record in frame_records:
            for key in record:
                if key not in frame_fieldnames:
                    frame_fieldnames.append(key)
        with (args.out_dir / "frame_diagnostics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=frame_fieldnames)
            writer.writeheader()
            writer.writerows(frame_records)

    if cluster_records:
        cluster_fieldnames = []
        for record in cluster_records:
            for key in record:
                if key not in cluster_fieldnames:
                    cluster_fieldnames.append(key)
        with (args.out_dir / "cluster_records.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cluster_fieldnames)
            writer.writeheader()
            writer.writerows(cluster_records)

    print(f"{'method':24s} {'WPC%':>7s} {'warn%':>7s} {'obj/fr':>7s} {'miss/fr':>8s} {'mRec':>7s} {'src':>5s}")
    for row in rows:
        print(
            f"{row['method']:24s} {100*row['WPC']:6.3f}% {100*row['warn_rate']:6.2f}% "
            f"{row['avg_obj_removed']:7.2f} {row['avg_missing_recovered']:8.2f} "
            f"{100*row['missing_recall']:6.2f}% {row['avg_sources_available']:5.2f}"
        )
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    if frame_records:
        print(f"Wrote {args.out_dir / 'frame_diagnostics.csv'}")
    if cluster_records:
        print(f"Wrote {args.out_dir / 'cluster_records.csv'}")


if __name__ == "__main__":
    main()

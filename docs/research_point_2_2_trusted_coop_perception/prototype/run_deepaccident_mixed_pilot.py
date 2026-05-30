#!/usr/bin/env python3
"""Run mixed-anomaly TrustCalib pilots on a DeepAccident subset."""
from __future__ import annotations

import argparse
import copy
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

from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.anomaly_injection import inject_anomaly
from prototype.trust_calib import (
    CalibReport,
    OffsetEstimate,
    calibrate_shifted_coop,
    filter_visible,
    greedy_associate,
    merge_ego_visible_with_coop_invisible,
)


def load_hybrid(use_v1: bool) -> HybridSafetyConstraint:
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


def eval_method(hybrid, waypoints, perception):
    modified, stats = hybrid.constrain_waypoints(waypoints, perception)
    warned = (
        stats.get("n_collisions_detected", 0) > 0
        or stats.get("modification_rate", 0) > 0
        or stats.get("n_geometric_threats", 0) > 0
    )
    return modified, stats, warned


def impact_guard_perception(
    ego_only,
    trust_perception,
    report: CalibReport,
    hybrid,
    waypoints,
    geom_delta_thr: float = 3.0,
    mod_delta_thr: float = 0.25,
):
    """Conservative safety-impact guard.

    If the cooperative message looks geometrically clean (accepted without
    correction) but greatly changes downstream geometric safety signals, treat it
    as unverified high-impact information and fall back to ego-only.

    This is intentionally conservative; the pilot should expose the trade-off.
    """
    if report.action != "accept":
        return trust_perception, report, False

    _, ego_stats, _ = eval_method(hybrid, waypoints, ego_only)
    _, trust_stats, _ = eval_method(hybrid, waypoints, trust_perception)
    geom_delta = trust_stats.get("n_geometric_threats", 0) - ego_stats.get("n_geometric_threats", 0)
    mod_delta = trust_stats.get("modification_rate", 0.0) - ego_stats.get("modification_rate", 0.0)
    if geom_delta >= geom_delta_thr or mod_delta >= mod_delta_thr:
        guarded = CalibReport(
            action="quarantine",
            message_usability=0.0,
            offset=report.offset,
            n_input_coop_agents=report.n_input_coop_agents,
            n_output_agents=len(ego_only.agents),
            n_corrected_agents=0,
        )
        return ego_only, guarded, True
    return trust_perception, report, False


def _agent_key(agent):
    return (
        agent.state.id,
        round(agent.state.x, 3),
        round(agent.state.y, 3),
        round(agent.state.length, 2),
        round(agent.state.width, 2),
    )


def _with_single_agent(base_perception, agent):
    p = copy.deepcopy(base_perception)
    p.agents.append(copy.deepcopy(agent))
    return p


class ObjectEvidenceTracker:
    """Minimal temporal support tracker for cooperative-only objects.

    The tracker only learns from objects that survive the object guard, plus
    ego-visible objects. A high-impact cooperative-only object that is removed
    on first sight therefore cannot become self-supporting merely by repeating
    the same unsupported claim in later frames.
    """

    def __init__(self, max_dist: float = 5.0):
        self.max_dist = max_dist
        self._tracks = []

    def is_supported(self, agent) -> bool:
        if agent.is_visible:
            return True
        agent_id = str(agent.state.id)
        for track in self._tracks:
            if track["id"] == agent_id:
                return True
        for track in self._tracks:
            if track["agent_type"] != agent.agent_type:
                continue
            dist = np.hypot(track["x"] - agent.state.x, track["y"] - agent.state.y)
            if dist <= self.max_dist:
                return True
        return False

    def update(self, ego_perception, kept_perception) -> None:
        evidence_agents = []
        evidence_agents.extend([a for a in ego_perception.agents if a.is_visible])
        evidence_agents.extend([a for a in kept_perception.agents if a.source == "coop"])
        tracks = []
        for agent in evidence_agents:
            tracks.append(
                {
                    "id": str(agent.state.id),
                    "agent_type": agent.agent_type,
                    "x": float(agent.state.x),
                    "y": float(agent.state.y),
                }
            )
        self._tracks = tracks


class PeerEvidenceSupport:
    """Object support from a second cooperative evidence source.

    This is a simulation harness: the current DeepAccident pilot has one object
    list, so `peer_oracle` uses the clean frame perception as if another sender
    corroborated the object. It should later be replaced by real multi-sender
    messages.
    """

    def __init__(self, peer_perception, max_dist: float = 2.0, max_size_diff: float = 3.0):
        self.peer_agents = list(peer_perception.agents)
        self.max_dist = max_dist
        self.max_size_diff = max_size_diff

    def is_supported(self, agent) -> bool:
        if agent.is_visible:
            return True
        for peer in self.peer_agents:
            if peer.agent_type != agent.agent_type:
                continue
            dist = np.hypot(peer.state.x - agent.state.x, peer.state.y - agent.state.y)
            if dist > self.max_dist:
                continue
            size = abs(peer.state.length - agent.state.length) + abs(peer.state.width - agent.state.width)
            if size <= self.max_size_diff:
                return True
        return False


def _has_object_evidence(agent) -> bool:
    """Placeholder for object-level evidence support.

    In a deployed system this should be replaced by temporal track continuity,
    peer corroboration, sender-side evidence chains, or raw-sensor support.
    In this synthetic pilot, `fake_front` is deliberately injected as an
    unsupported cooperative-only object.
    """
    if str(agent.state.id).startswith("fake_"):
        return False
    if agent.source == "coop_fake":
        return False
    return True


def object_impact_guard_perception(
    ego_perception,
    trust_perception,
    report: CalibReport,
    hybrid,
    waypoints,
    evidence_tracker: ObjectEvidenceTracker | None = None,
    geom_delta_thr: float = 3.0,
    mod_delta_thr: float = 0.25,
):
    """Remove only unsupported high-impact cooperative-only objects.

    A cooperative-only object is considered supported when the whole message was
    accepted/corrected with strong alignment evidence. This first version is
    intentionally conservative for accepted messages: clean messages retain most
    objects because each individual object's marginal impact is usually modest;
    fake_front is removed because it alone creates many waypoint threats.
    """
    if report.action in ("correct", "time_correct", "downweight"):
        return trust_perception, report, 0

    ego_visible = [a for a in ego_perception.agents if a.is_visible]
    coop_candidates = [a for a in trust_perception.agents if a.source == "coop"]
    supported_keys = set()
    if report.action == "accept" and report.offset.match_count >= 3:
        # Visible anchors support global source consistency. Object-level support
        # is represented here by track/evidence availability; fake injected
        # objects intentionally lack it.
        for coop in coop_candidates:
            has_evidence = (
                evidence_tracker.is_supported(coop)
                if evidence_tracker is not None
                else _has_object_evidence(coop)
            )
            if has_evidence:
                supported_keys.add(_agent_key(coop))
        for _, coop, _ in greedy_associate(ego_visible, coop_candidates, max_dist=2.0):
            supported_keys.add(_agent_key(coop))

    _, ego_stats, _ = eval_method(hybrid, waypoints, ego_perception)
    kept = []
    removed = 0
    for agent in trust_perception.agents:
        if agent.source != "coop":
            kept.append(agent)
            continue
        if _agent_key(agent) in supported_keys:
            kept.append(agent)
            continue

        single = _with_single_agent(ego_perception, agent)
        _, single_stats, _ = eval_method(hybrid, waypoints, single)
        geom_delta = single_stats.get("n_geometric_threats", 0) - ego_stats.get("n_geometric_threats", 0)
        mod_delta = single_stats.get("modification_rate", 0.0) - ego_stats.get("modification_rate", 0.0)
        if geom_delta >= geom_delta_thr or mod_delta >= mod_delta_thr:
            removed += 1
        else:
            kept.append(agent)

    guarded = copy.deepcopy(trust_perception)
    guarded.agents = kept
    if removed <= 0:
        return guarded, report, 0

    guarded_report = CalibReport(
        action="object_quarantine",
        message_usability=max(0.0, report.message_usability * (len(kept) / max(len(trust_perception.agents), 1))),
        offset=report.offset,
        n_input_coop_agents=report.n_input_coop_agents,
        n_output_agents=len(guarded.agents),
        n_corrected_agents=max(0, report.n_corrected_agents - removed),
    )
    return guarded, guarded_report, removed


def init_metrics(methods):
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
            "time_correct": 0,
            "downweight": 0,
            "quarantine": 0,
            "object_quarantine": 0,
            "dx_sum": 0.0,
            "dy_sum": 0.0,
            "before_sum": 0.0,
            "after_sum": 0.0,
            "corr_sum": 0.0,
            "match_sum": 0.0,
            "obj_removed_sum": 0,
            "fake_seen": 0,
            "fake_removed": 0,
        }
        for name in methods
    }


def update(
    metrics,
    name,
    frame,
    modified,
    stats,
    warned,
    report=None,
    obj_removed: int = 0,
    fake_seen: int = 0,
    fake_removed: int = 0,
):
    m = metrics[name]
    m["frames"] += 1
    m["warned"] += int(warned)
    m["wp_coll"] += check_waypoint_collision(modified, frame)
    m["wp_total"] += len(modified)
    m["mod_sum"] += stats.get("modification_rate", 0.0)
    m["geom_sum"] += stats.get("n_geometric_threats", 0)
    m["p_sum"] += stats.get("collision_prob", 0.0)
    m["obj_removed_sum"] += obj_removed
    m["fake_seen"] += fake_seen
    m["fake_removed"] += fake_removed
    if report is not None:
        m["reports"] += 1
        if report.action in ("accept", "correct", "time_correct", "downweight", "quarantine", "object_quarantine"):
            m[report.action] += 1
        m["dx_sum"] += report.offset.dx
        m["dy_sum"] += report.offset.dy
        m["before_sum"] += report.offset.residual_before
        m["after_sum"] += report.offset.residual_after
        m["corr_sum"] += report.offset.correctable_score
        m["match_sum"] += report.offset.match_count


def rows_from_metrics(mode, metrics):
    rows = []
    for name, m in metrics.items():
        frames = max(m["frames"], 1)
        reports = max(m["reports"], 1)
        rows.append(
            {
                "mode": mode,
                "method": name,
                "frames": m["frames"],
                "WPC": m["wp_coll"] / max(m["wp_total"], 1),
                "warn_rate": m["warned"] / frames,
                "avg_mod": m["mod_sum"] / frames,
                "avg_geom": m["geom_sum"] / frames,
                "avg_p_coll": m["p_sum"] / frames,
                "reports": m["reports"],
                "accept_rate": m["accept"] / reports if m["reports"] else 0.0,
                "correct_rate": m["correct"] / reports if m["reports"] else 0.0,
                "time_correct_rate": m["time_correct"] / reports if m["reports"] else 0.0,
                "downweight_rate": m["downweight"] / reports if m["reports"] else 0.0,
                "quarantine_rate": m["quarantine"] / reports if m["reports"] else 0.0,
                "object_quarantine_rate": m["object_quarantine"] / reports if m["reports"] else 0.0,
                "avg_dx": m["dx_sum"] / reports if m["reports"] else 0.0,
                "avg_dy": m["dy_sum"] / reports if m["reports"] else 0.0,
                "avg_residual_before": m["before_sum"] / reports if m["reports"] else 0.0,
                "avg_residual_after": m["after_sum"] / reports if m["reports"] else 0.0,
                "avg_correctable": m["corr_sum"] / reports if m["reports"] else 0.0,
                "avg_match_count": m["match_sum"] / reports if m["reports"] else 0.0,
                "avg_obj_removed": m["obj_removed_sum"] / frames,
                "fake_seen": m["fake_seen"],
                "fake_removed": m["fake_removed"],
                "fake_removal_rate": m["fake_removed"] / max(m["fake_seen"], 1),
            }
        )
    return rows


def count_fake_front(perception) -> int:
    return sum(1 for agent in perception.agents if str(agent.state.id) == "fake_front")


def run_mode(args, mode, loader, val_idx, hybrid):
    rng = np.random.default_rng(args.seed)
    methods = [
        "EgoOnly",
        "CleanCoop",
        "RawAnomalyCoop",
        "HardFilter",
        "TrustCalib",
        "TrustCalib+ImpactGuard",
        "TrustCalib+ObjectGuard",
    ]
    metrics = init_metrics(methods)

    for si in val_idx:
        evidence_tracker = ObjectEvidenceTracker() if args.evidence_mode == "temporal" else None
        scenario = loader.scenarios[si]
        n_frames = len(scenario["frames"])
        if args.max_frames_per_scenario > 0:
            n_frames = min(n_frames, args.max_frames_per_scenario)
        for fi in range(n_frames):
            frame = loader.load_frame(si, fi)
            waypoints = simulate_codriving_waypoints(frame)

            ego_only = filter_visible(frame.perception)
            clean_coop = merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)
            anomaly_msg = inject_anomaly(
                frame.perception,
                mode=mode,
                rng=rng,
                shift_x=args.shift_x,
                shift_y=args.shift_y,
                noise_sigma=args.noise_sigma,
                drop_rate=args.drop_rate,
                stale_delay_s=args.stale_delay_s,
            )
            raw_anomaly = merge_ego_visible_with_coop_invisible(frame.perception, anomaly_msg)
            trust_calib, report = calibrate_shifted_coop(frame.perception, anomaly_msg)
            guarded_calib, guarded_report, _ = impact_guard_perception(
                ego_only, trust_calib, report, hybrid, waypoints
            )
            frame_evidence = evidence_tracker
            if args.evidence_mode == "peer_oracle":
                frame_evidence = PeerEvidenceSupport(frame.perception)
            object_guarded, object_report, obj_removed = object_impact_guard_perception(
                ego_only, trust_calib, report, hybrid, waypoints, evidence_tracker=frame_evidence
            )
            fake_seen = count_fake_front(trust_calib)
            fake_removed = max(0, fake_seen - count_fake_front(object_guarded))

            per_method = {
                "EgoOnly": (ego_only, None, 0, 0, 0),
                "CleanCoop": (clean_coop, None, 0, 0, 0),
                "RawAnomalyCoop": (raw_anomaly, None, 0, 0, 0),
                "HardFilter": (ego_only, None, 0, 0, 0),
                "TrustCalib": (trust_calib, report, 0, 0, 0),
                "TrustCalib+ImpactGuard": (guarded_calib, guarded_report, 0, 0, 0),
                "TrustCalib+ObjectGuard": (
                    object_guarded,
                    object_report,
                    obj_removed,
                    fake_seen,
                    fake_removed,
                ),
            }
            for name, (perception, rep, removed, fake_n, fake_removed_n) in per_method.items():
                modified, stats, warned = eval_method(hybrid, waypoints, perception)
                update(
                    metrics,
                    name,
                    frame,
                    modified,
                    stats,
                    warned,
                    rep,
                    obj_removed=removed,
                    fake_seen=fake_n,
                    fake_removed=fake_removed_n,
                )
            if evidence_tracker is not None:
                evidence_tracker.update(ego_only, object_guarded)

    return rows_from_metrics(mode, metrics)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="clean,shift,shift_severe,noise,drop,stale,fake_front")
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--max-frames-per-scenario", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shift-x", type=float, default=2.0)
    parser.add_argument("--shift-y", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--drop-rate", type=float, default=0.7)
    parser.add_argument("--stale-delay-s", type=float, default=1.0)
    parser.add_argument(
        "--evidence-mode",
        choices=("synthetic", "temporal", "peer_oracle"),
        default="synthetic",
        help="object evidence source for ObjectGuard",
    )
    parser.add_argument("--disable-v1", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/deepaccident_mixed_pilot"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    hybrid = load_hybrid(use_v1=not args.disable_v1)
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=False)
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]

    all_rows = []
    for mode in modes:
        print(f"Running mode={mode}")
        rows = run_mode(args, mode, loader, val_idx, hybrid)
        all_rows.extend(rows)

    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    meta = vars(args).copy()
    meta["modes"] = modes
    meta["val_scenarios_used"] = len(val_idx)
    meta["disable_v1"] = bool(args.disable_v1)
    meta["out_dir"] = str(args.out_dir)
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'mode':13s} {'method':24s} {'WPC%':>7s} {'warn%':>7s} {'act':>6s} {'corr':>6s} {'down':>6s} {'quar':>6s} {'objQ':>6s} {'rm/fr':>8s} {'dx':>7s} {'dy':>7s}")
    for row in all_rows:
        if row["method"] not in (
            "CleanCoop",
            "RawAnomalyCoop",
            "HardFilter",
            "TrustCalib",
            "TrustCalib+ImpactGuard",
            "TrustCalib+ObjectGuard",
        ):
            continue
        print(
            f"{row['mode']:13s} {row['method']:24s} {100*row['WPC']:6.2f}% "
            f"{100*row['warn_rate']:6.2f}% {100*row['accept_rate']:5.1f}% "
            f"{100*row['correct_rate']:5.1f}% {100*row['downweight_rate']:5.1f}% "
            f"{100*row['quarantine_rate']:5.1f}% {100*row['object_quarantine_rate']:5.1f}% "
            f"{row['avg_obj_removed']:4.2f}/{100*row['fake_removal_rate']:3.0f}% "
            f"{row['avg_dx']:7.2f} {row['avg_dy']:7.2f}"
        )
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()

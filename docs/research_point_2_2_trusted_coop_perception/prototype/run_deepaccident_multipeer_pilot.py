#!/usr/bin/env python3
"""Run explicit multi-peer evidence pilots on a DeepAccident subset."""
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

from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import check_waypoint_collision, simulate_codriving_waypoints
from prototype.anomaly_injection import inject_anomaly
from prototype.run_deepaccident_mixed_pilot import (
    eval_method,
    load_hybrid,
    object_impact_guard_perception,
)
from prototype.trust_calib import (
    calibrate_shifted_coop,
    calibrate_temporal_coop,
    filter_visible,
    merge_ego_visible_with_coop_invisible,
)


class MultiPeerEvidenceSupport:
    """Cluster support from calibrated peer messages.

    For a primary object, its evidence cluster is the set of matching
    cooperative objects from support peers. The cluster exposes count, weighted
    support, spatial spread, and offset consistency for interpretability.
    """

    def __init__(
        self,
        peer_perceptions,
        peer_reports=None,
        sender_ids=None,
        sender_trusts=None,
        min_support: int = 1,
        min_trust_support: float = 0.0,
        max_dist: float = 2.0,
        max_size_diff: float = 3.0,
    ):
        self.min_support = min_support
        self.min_trust_support = min_trust_support
        self.max_dist = max_dist
        self.max_size_diff = max_size_diff
        self.peer_reports = peer_reports or [None] * len(peer_perceptions)
        self.sender_ids = sender_ids or [f"peer_{i}" for i in range(len(peer_perceptions))]
        self.sender_trusts = sender_trusts or [1.0] * len(peer_perceptions)
        self.peer_agent_sets = [
            [agent for agent in perception.agents if agent.source == "coop"]
            for perception in peer_perceptions
        ]

    def is_supported(self, agent) -> bool:
        details = self.cluster_details(agent)
        return (
            details["support_count"] >= self.min_support
            and details["trust_weighted_support"] >= self.min_trust_support
        )

    def support_count(self, agent) -> int:
        return self.cluster_details(agent)["support_count"]

    def cluster_details(self, agent) -> dict:
        if agent.is_visible:
            return {
                "support_count": len(self.peer_agent_sets),
                "supporting_sender_ids": list(self.sender_ids),
                "trust_weighted_support": float(sum(self.sender_trusts)),
                "position_cov_trace": 0.0,
                "offset_spread": 0.0,
            }

        support = 0
        support_weight = 0.0
        senders = []
        positions = [[float(agent.state.x), float(agent.state.y)]]
        offsets = []
        for i, peer_agents in enumerate(self.peer_agent_sets):
            match = self._best_match(agent, peer_agents)
            if match is not None:
                support += 1
                support_weight += float(self.sender_trusts[i])
                senders.append(self.sender_ids[i])
                positions.append([float(match.state.x), float(match.state.y)])
                report = self.peer_reports[i] if i < len(self.peer_reports) else None
                if report is not None:
                    offsets.append([float(report.offset.dx), float(report.offset.dy)])

        pos_cov_trace = 0.0
        if len(positions) >= 2:
            arr = np.array(positions, dtype=float)
            pos_cov_trace = float(np.trace(np.cov(arr.T)))

        offset_spread = 0.0
        if len(offsets) >= 2:
            arr = np.array(offsets, dtype=float)
            center = np.mean(arr, axis=0)
            offset_spread = float(np.mean(np.linalg.norm(arr - center, axis=1)))

        return {
            "support_count": support,
            "supporting_sender_ids": senders,
            "trust_weighted_support": support_weight,
            "position_cov_trace": pos_cov_trace,
            "offset_spread": offset_spread,
        }

    def support_matches(self, agent):
        matches = []
        for i, peer_agents in enumerate(self.peer_agent_sets):
            match = self._best_match(agent, peer_agents)
            if match is not None:
                matches.append(
                    {
                        "peer_index": i,
                        "sender_id": self.sender_ids[i],
                        "sender_trust": float(self.sender_trusts[i]),
                        "agent": match,
                    }
                )
        return matches

    def _best_match(self, agent, peer_agents):
        best = None
        best_cost = float("inf")
        for peer in peer_agents:
            if peer.agent_type != agent.agent_type:
                continue
            dist = np.hypot(peer.state.x - agent.state.x, peer.state.y - agent.state.y)
            if dist > self.max_dist:
                continue
            size = abs(peer.state.length - agent.state.length) + abs(peer.state.width - agent.state.width)
            if size > self.max_size_diff:
                continue
            cost = float(dist + 0.5 * size)
            if cost < best_cost:
                best = peer
                best_cost = cost
        return best


def count_fake(perception) -> int:
    return sum(1 for agent in perception.agents if str(agent.state.id) == "fake_front")


def agent_key(agent):
    return (
        agent.state.id,
        round(agent.state.x, 3),
        round(agent.state.y, 3),
        round(agent.state.length, 2),
        round(agent.state.width, 2),
    )


class UnsupportedTemporalTracker:
    """Track unsupported object age without turning age into evidence support."""

    def __init__(self, max_dist: float = 5.0):
        self.max_dist = max_dist
        self._tracks = []

    def update_and_label(self, primary_perception, evidence) -> dict:
        labels = {}
        next_tracks = []
        used_prev = set()
        for agent in primary_perception.agents:
            if agent.source != "coop":
                continue
            key = agent_key(agent)
            if evidence.is_supported(agent):
                labels[key] = {"temporal_status": "supported", "unsupported_age": 0}
                continue

            prev_idx = self._match_previous(agent, used_prev)
            if prev_idx is None:
                age = 1
                status = "new_unsupported"
            else:
                age = self._tracks[prev_idx]["age"] + 1
                status = "persistent_unsupported"
                used_prev.add(prev_idx)

            labels[key] = {"temporal_status": status, "unsupported_age": age}
            next_tracks.append(
                {
                    "id": str(agent.state.id),
                    "agent_type": agent.agent_type,
                    "x": float(agent.state.x),
                    "y": float(agent.state.y),
                    "age": age,
                }
            )
        self._tracks = next_tracks
        return labels

    def _match_previous(self, agent, used_prev):
        agent_id = str(agent.state.id)
        for i, track in enumerate(self._tracks):
            if i in used_prev:
                continue
            if track["id"] == agent_id:
                return i
        best_i = None
        best_dist = float("inf")
        for i, track in enumerate(self._tracks):
            if i in used_prev or track["agent_type"] != agent.agent_type:
                continue
            dist = float(np.hypot(track["x"] - agent.state.x, track["y"] - agent.state.y))
            if dist <= self.max_dist and dist < best_dist:
                best_i = i
                best_dist = dist
        return best_i


class SenderTrustState:
    """Simple sender trust dynamics driven by object-level evidence."""

    def __init__(self, initial_trust: float, min_trust: float = 0.0, max_trust: float = 1.0):
        self.trust = float(initial_trust)
        self.min_trust = min_trust
        self.max_trust = max_trust

    def update(
        self,
        records,
        penalty: float,
        reward: float,
        offset_instability: float = 0.0,
        peer_disagreement: float = 0.0,
        offset_penalty: float = 0.0,
        disagreement_penalty: float = 0.0,
        offset_thr: float = 0.5,
        disagreement_thr: float = 1.0,
    ) -> float:
        high_impact_unsupported = 0
        supported = 0
        for record in records:
            high_impact = (
                float(record["geom_delta"]) > 2.0
                or float(record["mod_delta"]) > 0.2
            )
            if int(record["evidence_supported"]):
                supported += 1
            elif high_impact:
                high_impact_unsupported += 1

        self.trust -= penalty * high_impact_unsupported
        if offset_instability > offset_thr:
            self.trust -= offset_penalty * (offset_instability - offset_thr)
        if peer_disagreement > disagreement_thr:
            self.trust -= disagreement_penalty * (peer_disagreement - disagreement_thr)
        if high_impact_unsupported == 0 and supported > 0:
            self.trust += reward
        self.trust = float(np.clip(self.trust, self.min_trust, self.max_trust))
        return self.trust


def count_real_removed(before, after) -> int:
    after_agents = [agent for agent in after.agents if agent.source == "coop"]
    removed = 0
    for agent in before.agents:
        if agent.source != "coop":
            continue
        if str(agent.state.id) == "fake_front":
            continue
        if _geometry_match(agent, after_agents, max_dist=4.0, max_size_diff=3.0) is None:
            removed += 1
    return removed


def _agent_marginal_impact(agent, ego_only, hybrid, waypoints):
    _, ego_stats, _ = eval_method(hybrid, waypoints, ego_only)
    _, single_stats, _ = eval_method(hybrid, waypoints, single_agent_perception(ego_only, agent))
    geom_delta = single_stats.get("n_geometric_threats", 0) - ego_stats.get("n_geometric_threats", 0)
    mod_delta = single_stats.get("modification_rate", 0.0) - ego_stats.get("modification_rate", 0.0)
    return geom_delta, mod_delta


def apply_high_trust_probation(primary_perception, guarded_perception, temporal_labels, args, ego_only, hybrid, waypoints):
    if not args.enable_high_trust_probation:
        return guarded_perception, 0, {}
    if args.primary_trust < args.probation_primary_trust_thr:
        return guarded_perception, 0, {}

    restored = copy.deepcopy(guarded_perception)
    kept_keys = {agent_key(agent) for agent in restored.agents if agent.source == "coop"}
    n_restored = 0
    restored_reason = {}
    for agent in primary_perception.agents:
        if agent.source != "coop":
            continue
        key = agent_key(agent)
        if key in kept_keys:
            continue
        temporal = temporal_labels.get(key, {"temporal_status": "unknown", "unsupported_age": 999})
        if temporal["temporal_status"] not in ("new_unsupported", "persistent_unsupported"):
            continue
        if temporal["unsupported_age"] > args.probation_max_age:
            continue
        geom_delta, mod_delta = _agent_marginal_impact(agent, ego_only, hybrid, waypoints)
        high_impact = (
            geom_delta > args.probation_max_geom_delta
            or mod_delta > args.probation_max_mod_delta
        )
        if high_impact and not args.probation_allow_high_impact:
            continue
        restored.agents.append(copy.deepcopy(agent))
        kept_keys.add(key)
        n_restored += 1
        restored_reason[key] = "high_trust_probation_high_impact" if high_impact else "high_trust_probation_low_impact"
    return restored, n_restored, restored_reason


def _geometry_match(agent, candidates, max_dist: float, max_size_diff: float):
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


def _represent_missing_cluster(cluster, sender_trusts):
    members = cluster["members"]
    weights = np.array([max(float(sender_trusts[peer_idx]), 1e-6) for peer_idx, _ in members], dtype=float)
    agents = [agent for _, agent in members]
    rep = copy.deepcopy(agents[0])
    rep.state.x = float(np.average([a.state.x for a in agents], weights=weights))
    rep.state.y = float(np.average([a.state.y for a in agents], weights=weights))
    rep.state.vx = float(np.average([a.state.vx for a in agents], weights=weights))
    rep.state.vy = float(np.average([a.state.vy for a in agents], weights=weights))
    rep.state.length = float(np.average([a.state.length for a in agents], weights=weights))
    rep.state.width = float(np.average([a.state.width for a in agents], weights=weights))
    rep.is_visible = False
    rep.source = "coop"
    rep.confidence = float(np.clip(np.average([a.confidence for a in agents], weights=weights), 0.0, 1.0))
    return rep


def build_missing_recovery_candidates(primary_perception, support_perceptions, sender_ids, sender_trusts, args):
    primary_agents = list(primary_perception.agents)
    clusters = []
    for peer_idx, support in enumerate(support_perceptions):
        support_agents = [agent for agent in support.agents if agent.source == "coop"]
        for agent in support_agents:
            if _geometry_match(agent, primary_agents, args.missing_match_dist, args.missing_max_size_diff) is not None:
                continue

            best_i = None
            best_dist = float("inf")
            for i, cluster in enumerate(clusters):
                if peer_idx in cluster["peer_indices"]:
                    continue
                center = cluster["center"]
                if center.agent_type != agent.agent_type:
                    continue
                dist = float(np.hypot(center.state.x - agent.state.x, center.state.y - agent.state.y))
                if dist > args.missing_cluster_dist:
                    continue
                size = abs(center.state.length - agent.state.length) + abs(center.state.width - agent.state.width)
                if size > args.missing_max_size_diff:
                    continue
                if dist < best_dist:
                    best_i = i
                    best_dist = dist

            if best_i is None:
                clusters.append(
                    {
                        "members": [(peer_idx, agent)],
                        "peer_indices": {peer_idx},
                        "center": copy.deepcopy(agent),
                    }
                )
            else:
                cluster = clusters[best_i]
                cluster["members"].append((peer_idx, agent))
                cluster["peer_indices"].add(peer_idx)
                cluster["center"] = _represent_missing_cluster(cluster, sender_trusts)

    candidates = []
    for cluster_id, cluster in enumerate(clusters):
        support_count = len(cluster["peer_indices"])
        trust_weighted_support = float(sum(sender_trusts[i] for i in cluster["peer_indices"]))
        if support_count < args.missing_min_peer_support:
            continue
        if trust_weighted_support < args.missing_min_trust_support:
            continue

        representative = _represent_missing_cluster(cluster, sender_trusts)
        positions = np.array(
            [[agent.state.x, agent.state.y] for _, agent in cluster["members"]],
            dtype=float,
        )
        if len(positions) >= 2:
            cov_trace = float(np.trace(np.cov(positions.T)))
        else:
            cov_trace = 0.0
        candidates.append(
            {
                "cluster_id": cluster_id,
                "agent": representative,
                "support_count": support_count,
                "supporting_sender_ids": [sender_ids[i] for i in sorted(cluster["peer_indices"])],
                "trust_weighted_support": trust_weighted_support,
                "position_cov_trace": cov_trace,
                "offset_spread": 0.0,
            }
        )
    return candidates


def missing_reference_agents(reference_perception, primary_perception, args):
    primary_agents = list(primary_perception.agents)
    missing = []
    for agent in reference_perception.agents:
        if agent.source != "coop":
            continue
        if _geometry_match(agent, primary_agents, args.missing_match_dist, args.missing_max_size_diff) is None:
            missing.append(agent)
    return missing


def apply_missing_object_recovery(
    mode,
    scenario_index,
    frame_index,
    guarded_perception,
    primary_perception,
    primary_report,
    reference_perception,
    support_perceptions,
    sender_ids,
    sender_trusts,
    args,
    ego_only,
    hybrid,
    waypoints,
):
    if not args.enable_missing_recovery:
        return guarded_perception, 0, 0, 0, 0, 0, []
    missing_gt = missing_reference_agents(reference_perception, primary_perception, args)
    allowed_actions = {
        action.strip()
        for action in args.missing_recovery_primary_actions.split(",")
        if action.strip()
    }
    if primary_report.action not in allowed_actions:
        return guarded_perception, 0, 0, len(missing_gt), 0, 0, []

    candidates = build_missing_recovery_candidates(
        primary_perception,
        support_perceptions,
        sender_ids,
        sender_trusts,
        args,
    )
    recovered = copy.deepcopy(guarded_perception)
    existing_agents = list(recovered.agents)
    records = []
    n_recovered = 0
    n_recovery_tp = 0
    n_recovery_fp = 0
    matched_gt_indices = set()
    for candidate in candidates:
        agent = candidate["agent"]
        if _geometry_match(agent, existing_agents, args.missing_match_dist, args.missing_max_size_diff) is not None:
            continue

        geom_delta, mod_delta = _agent_marginal_impact(agent, ego_only, hybrid, waypoints)
        path_min_distance, path_collision_margin, path_risk_step = path_margin_features(waypoints, agent)
        path_box_distance, path_box_collision_margin, path_box_risk_step = path_oriented_box_margin_features(
            waypoints,
            agent,
            ego_radius=args.path_box_ego_radius,
        )
        high_impact = (
            geom_delta >= args.missing_impact_geom_thr
            or mod_delta >= args.missing_impact_mod_thr
        )
        final_action = "missing_skip_low_impact"
        recovery_eval = "not_recovered"
        matched_reference_id = ""
        if high_impact or not args.missing_recover_high_impact_only:
            recovered_agent = copy.deepcopy(agent)
            recovered_agent.source = "coop"
            recovered_agent.is_visible = False
            recovered.agents.append(recovered_agent)
            existing_agents.append(recovered_agent)
            n_recovered += 1
            final_action = "missing_recover_high_impact" if high_impact else "missing_recover"
            match_idx = None
            match = _geometry_match(agent, missing_gt, args.missing_match_dist, args.missing_max_size_diff)
            if match is not None:
                for i, gt_agent in enumerate(missing_gt):
                    if gt_agent is match:
                        match_idx = i
                        break
            if match_idx is not None:
                if match_idx not in matched_gt_indices:
                    matched_gt_indices.add(match_idx)
                    n_recovery_tp += 1
                    recovery_eval = "tp"
                    matched_reference_id = str(match.state.id)
                else:
                    n_recovery_fp += 1
                    recovery_eval = "duplicate_fp"
                    matched_reference_id = str(match.state.id)
            else:
                n_recovery_fp += 1
                recovery_eval = "fp"

        distance, ttc, closest_distance = kinematic_impact_features(ego_only.ego, agent)
        records.append(
            {
                "mode": mode,
                "scenario_index": scenario_index,
                "frame_index": frame_index,
                "primary_sender": "missing_recovery",
                "object_id": f"missing_cluster_{candidate['cluster_id']}_{agent.state.id}",
                "is_fake": 0,
                "x": float(agent.state.x),
                "y": float(agent.state.y),
                "evidence_supported": 1,
                "temporal_status": "peer_missing_candidate",
                "unsupported_age": 0,
                "support_count": candidate["support_count"],
                "supporting_sender_ids": ";".join(candidate["supporting_sender_ids"]),
                "trust_weighted_support": candidate["trust_weighted_support"],
                "position_cov_trace": candidate["position_cov_trace"],
                "offset_spread": candidate["offset_spread"],
                "distance": distance,
                "ttc_s": ttc,
                "closest_distance": closest_distance,
                "path_min_distance": path_min_distance,
                "path_collision_margin": path_collision_margin,
                "path_risk_step": path_risk_step,
                "path_box_distance": path_box_distance,
                "path_box_collision_margin": path_box_collision_margin,
                "path_box_risk_step": path_box_risk_step,
                "geom_delta": geom_delta,
                "mod_delta": mod_delta,
                "final_action": final_action,
                "missing_recovery_eval": recovery_eval,
                "matched_reference_id": matched_reference_id,
            }
        )
    return recovered, n_recovered, len(candidates), len(missing_gt), n_recovery_tp, n_recovery_fp, records


def apply_box_margin_guard(guarded_perception, evidence, args, waypoints):
    if not args.enable_box_margin_guard:
        return guarded_perception, 0

    filtered = copy.deepcopy(guarded_perception)
    kept = []
    removed = 0
    for agent in filtered.agents:
        if agent.source != "coop":
            kept.append(agent)
            continue
        if evidence.is_supported(agent):
            kept.append(agent)
            continue
        _, box_margin, _ = path_oriented_box_margin_features(
            waypoints,
            agent,
            ego_radius=args.path_box_ego_radius,
        )
        if box_margin < args.box_margin_guard_thr:
            removed += 1
        else:
            kept.append(agent)
    filtered.agents = kept
    return filtered, removed


def apply_peer_consensus_smoothing(primary_perception, support_perceptions, sender_ids, sender_trusts, args):
    """Smooth cooperative object states with agreement from calibrated peers."""
    if not args.enable_peer_consensus_smoothing:
        return primary_perception, 0, 0.0

    smoothing_evidence = MultiPeerEvidenceSupport(
        support_perceptions,
        sender_ids=sender_ids,
        sender_trusts=sender_trusts,
        min_support=args.smooth_min_peer_support,
        min_trust_support=args.smooth_min_trust_support,
        max_dist=args.smooth_match_dist,
        max_size_diff=args.smooth_max_size_diff,
    )
    smoothed = copy.deepcopy(primary_perception)
    n_smoothed = 0
    residual_sum = 0.0

    for agent in smoothed.agents:
        if agent.source != "coop" or agent.is_visible:
            continue
        details = smoothing_evidence.cluster_details(agent)
        if details["support_count"] < args.smooth_min_peer_support:
            continue
        if details["trust_weighted_support"] < args.smooth_min_trust_support:
            continue
        if details["position_cov_trace"] > args.smooth_max_cluster_cov:
            continue

        matches = smoothing_evidence.support_matches(agent)
        if not matches:
            continue
        weights = np.array([max(match["sender_trust"], 1e-6) for match in matches], dtype=float)
        support_agents = [match["agent"] for match in matches]
        center_x = float(np.average([peer.state.x for peer in support_agents], weights=weights))
        center_y = float(np.average([peer.state.y for peer in support_agents], weights=weights))
        residual = float(np.hypot(agent.state.x - center_x, agent.state.y - center_y))
        if residual < args.smooth_min_primary_residual:
            continue

        alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
        agent.state.x = (1.0 - alpha) * agent.state.x + alpha * center_x
        agent.state.y = (1.0 - alpha) * agent.state.y + alpha * center_y
        agent.state.vx = (1.0 - alpha) * agent.state.vx + alpha * float(
            np.average([peer.state.vx for peer in support_agents], weights=weights)
        )
        agent.state.vy = (1.0 - alpha) * agent.state.vy + alpha * float(
            np.average([peer.state.vy for peer in support_agents], weights=weights)
        )
        agent.state.heading = (1.0 - alpha) * agent.state.heading + alpha * float(
            np.average([peer.state.heading for peer in support_agents], weights=weights)
        )
        agent.state.length = (1.0 - alpha) * agent.state.length + alpha * float(
            np.average([peer.state.length for peer in support_agents], weights=weights)
        )
        agent.state.width = (1.0 - alpha) * agent.state.width + alpha * float(
            np.average([peer.state.width for peer in support_agents], weights=weights)
        )
        agent.confidence = float(np.clip(max(agent.confidence, np.mean(weights)), 0.0, 1.0))
        n_smoothed += 1
        residual_sum += residual

    return smoothed, n_smoothed, residual_sum


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
            "probation_restore": 0,
            "dx_sum": 0.0,
            "dy_sum": 0.0,
            "obj_removed_sum": 0,
            "fake_seen": 0,
            "fake_removed": 0,
            "real_removed": 0,
            "real_support_sum": 0.0,
            "real_support_n": 0,
            "fake_support_sum": 0.0,
            "fake_support_n": 0,
            "real_trust_support_sum": 0.0,
            "fake_trust_support_sum": 0.0,
            "real_cov_sum": 0.0,
            "fake_cov_sum": 0.0,
            "real_offset_spread_sum": 0.0,
            "fake_offset_spread_sum": 0.0,
            "missing_candidate_sum": 0,
            "missing_recovered_sum": 0,
            "missing_gt_sum": 0,
            "missing_recovery_tp_sum": 0,
            "missing_recovery_fp_sum": 0,
            "delay_sum": 0.0,
            "smooth_agent_sum": 0,
            "smooth_residual_sum": 0.0,
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
    obj_removed=0,
    fake_seen=0,
    fake_removed=0,
    real_removed=0,
    real_support_sum=0.0,
    real_support_n=0,
    fake_support_sum=0.0,
    fake_support_n=0,
    real_trust_support_sum=0.0,
    fake_trust_support_sum=0.0,
    real_cov_sum=0.0,
    fake_cov_sum=0.0,
    real_offset_spread_sum=0.0,
    fake_offset_spread_sum=0.0,
    missing_candidates=0,
    missing_recovered=0,
    missing_gt=0,
    missing_recovery_tp=0,
    missing_recovery_fp=0,
    smooth_agents=0,
    smooth_residual=0.0,
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
    m["real_removed"] += real_removed
    m["real_support_sum"] += real_support_sum
    m["real_support_n"] += real_support_n
    m["fake_support_sum"] += fake_support_sum
    m["fake_support_n"] += fake_support_n
    m["real_trust_support_sum"] += real_trust_support_sum
    m["fake_trust_support_sum"] += fake_trust_support_sum
    m["real_cov_sum"] += real_cov_sum
    m["fake_cov_sum"] += fake_cov_sum
    m["real_offset_spread_sum"] += real_offset_spread_sum
    m["fake_offset_spread_sum"] += fake_offset_spread_sum
    m["missing_candidate_sum"] += missing_candidates
    m["missing_recovered_sum"] += missing_recovered
    m["missing_gt_sum"] += missing_gt
    m["missing_recovery_tp_sum"] += missing_recovery_tp
    m["missing_recovery_fp_sum"] += missing_recovery_fp
    m["smooth_agent_sum"] += smooth_agents
    m["smooth_residual_sum"] += smooth_residual
    if report is not None:
        m["reports"] += 1
        if report.action in (
            "accept",
            "correct",
            "time_correct",
            "downweight",
            "quarantine",
            "object_quarantine",
            "probation_restore",
        ):
            m[report.action] += 1
        m["dx_sum"] += report.offset.dx
        m["dy_sum"] += report.offset.dy
        m["delay_sum"] += getattr(report.offset, "delay_s", 0.0)


def rows_from_metrics(mode, support_modes, metrics):
    rows = []
    for name, m in metrics.items():
        frames = max(m["frames"], 1)
        reports = max(m["reports"], 1)
        rows.append(
            {
                "mode": mode,
                "support_modes": "+".join(support_modes),
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
                "time_correct_rate": m["time_correct"] / reports if m["reports"] else 0.0,
                "downweight_rate": m["downweight"] / reports if m["reports"] else 0.0,
                "quarantine_rate": m["quarantine"] / reports if m["reports"] else 0.0,
                "object_quarantine_rate": m["object_quarantine"] / reports if m["reports"] else 0.0,
                "probation_restore_rate": m["probation_restore"] / reports if m["reports"] else 0.0,
                "avg_dx": m["dx_sum"] / reports if m["reports"] else 0.0,
                "avg_dy": m["dy_sum"] / reports if m["reports"] else 0.0,
                "avg_delay_s": m["delay_sum"] / reports if m["reports"] else 0.0,
                "avg_obj_removed": m["obj_removed_sum"] / frames,
                "fake_seen": m["fake_seen"],
                "fake_removed": m["fake_removed"],
                "fake_removal_rate": m["fake_removed"] / max(m["fake_seen"], 1),
                "real_removed": m["real_removed"],
                "clean_false_remove_per_frame": m["real_removed"] / frames,
                "avg_real_peer_support": m["real_support_sum"] / max(m["real_support_n"], 1),
                "avg_fake_peer_support": m["fake_support_sum"] / max(m["fake_support_n"], 1),
                "avg_real_trust_support": m["real_trust_support_sum"] / max(m["real_support_n"], 1),
                "avg_fake_trust_support": m["fake_trust_support_sum"] / max(m["fake_support_n"], 1),
                "avg_real_cluster_cov": m["real_cov_sum"] / max(m["real_support_n"], 1),
                "avg_fake_cluster_cov": m["fake_cov_sum"] / max(m["fake_support_n"], 1),
                "avg_real_offset_spread": m["real_offset_spread_sum"] / max(m["real_support_n"], 1),
                "avg_fake_offset_spread": m["fake_offset_spread_sum"] / max(m["fake_support_n"], 1),
                "avg_missing_candidates": m["missing_candidate_sum"] / frames,
                "avg_missing_recovered": m["missing_recovered_sum"] / frames,
                "avg_missing_gt": m["missing_gt_sum"] / frames,
                "missing_recovery_tp": m["missing_recovery_tp_sum"],
                "missing_recovery_fp": m["missing_recovery_fp_sum"],
                "missing_recovery_precision": m["missing_recovery_tp_sum"] / max(m["missing_recovery_tp_sum"] + m["missing_recovery_fp_sum"], 1),
                "missing_recovery_recall": m["missing_recovery_tp_sum"] / max(m["missing_gt_sum"], 1),
                "missing_recovery_rate": m["missing_recovered_sum"] / max(m["missing_candidate_sum"], 1),
                "avg_smoothed_agents": m["smooth_agent_sum"] / frames,
                "avg_smooth_residual": m["smooth_residual_sum"] / max(m["smooth_agent_sum"], 1),
            }
        )
    return rows


def build_peer_messages(frame, mode, support_modes, rng, args):
    primary = inject_anomaly(
        frame.perception,
        mode=mode,
        rng=rng,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        noise_sigma=args.noise_sigma,
        drop_rate=args.drop_rate,
        stale_delay_s=args.stale_delay_s,
    )
    support = [
        inject_anomaly(
            frame.perception,
            mode=support_mode,
            rng=rng,
            shift_x=args.shift_x,
            shift_y=args.shift_y,
            noise_sigma=args.noise_sigma,
            drop_rate=args.drop_rate,
            stale_delay_s=args.stale_delay_s,
        )
        for support_mode in support_modes
    ]
    return primary, support


def parse_delay_grid(grid_text: str) -> list[float]:
    return [float(v.strip()) for v in grid_text.split(",") if v.strip()]


def calibrate_coop_message(ego_perception, coop_message, args):
    spatial_perception, spatial_report = calibrate_shifted_coop(ego_perception, coop_message)
    if not args.enable_time_calib:
        return spatial_perception, spatial_report

    temporal_perception, temporal_report = calibrate_temporal_coop(
        ego_perception,
        coop_message,
        delay_grid=args.time_delay_grid_values,
        correctable_thr=args.time_correctable_thr,
        downweight_thr=args.time_downweight_thr,
        clean_residual_thr=args.time_clean_residual_thr,
        min_delay_s=args.time_min_delay_s,
    )
    temporal_gain = spatial_report.offset.residual_after - temporal_report.offset.residual_after
    if (
        temporal_report.action == "time_correct"
        and temporal_report.offset.delay_s >= args.time_min_delay_s
        and temporal_gain >= args.time_min_residual_gain
    ):
        return temporal_perception, temporal_report
    return spatial_perception, spatial_report


def support_summary(primary_perception, evidence):
    real_sum = 0.0
    real_n = 0
    fake_sum = 0.0
    fake_n = 0
    real_trust_sum = 0.0
    fake_trust_sum = 0.0
    real_cov_sum = 0.0
    fake_cov_sum = 0.0
    real_offset_sum = 0.0
    fake_offset_sum = 0.0
    for agent in primary_perception.agents:
        if agent.source != "coop":
            continue
        details = evidence.cluster_details(agent)
        support = details["support_count"]
        if str(agent.state.id) == "fake_front":
            fake_sum += support
            fake_n += 1
            fake_trust_sum += details["trust_weighted_support"]
            fake_cov_sum += details["position_cov_trace"]
            fake_offset_sum += details["offset_spread"]
        else:
            real_sum += support
            real_n += 1
            real_trust_sum += details["trust_weighted_support"]
            real_cov_sum += details["position_cov_trace"]
            real_offset_sum += details["offset_spread"]
    return (
        real_sum,
        real_n,
        fake_sum,
        fake_n,
        real_trust_sum,
        fake_trust_sum,
        real_cov_sum,
        fake_cov_sum,
        real_offset_sum,
        fake_offset_sum,
    )


def single_agent_perception(base_perception, agent):
    perception = copy.deepcopy(base_perception)
    perception.agents.append(copy.deepcopy(agent))
    return perception


def kinematic_impact_features(ego, agent):
    rel_pos = np.array([agent.state.x - ego.x, agent.state.y - ego.y], dtype=float)
    rel_vel = np.array([agent.state.vx - ego.vx, agent.state.vy - ego.vy], dtype=float)
    distance = float(np.linalg.norm(rel_pos))
    speed_sq = float(np.dot(rel_vel, rel_vel))
    if speed_sq <= 1e-6:
        ttc = 999.0
        closest_distance = distance
    else:
        ttc_raw = -float(np.dot(rel_pos, rel_vel)) / speed_sq
        if ttc_raw < 0.0:
            ttc = 999.0
            closest_distance = distance
        else:
            ttc = min(ttc_raw, 999.0)
            closest = rel_pos + rel_vel * ttc
            closest_distance = float(np.linalg.norm(closest))
    return distance, ttc, closest_distance


def path_margin_features(waypoints, agent, collision_threshold: float = 2.0):
    min_dist = float("inf")
    min_step = -1
    for t, waypoint in enumerate(waypoints):
        dt = (t + 1) * 0.5
        ax = agent.state.x + agent.state.vx * dt
        ay = agent.state.y + agent.state.vy * dt
        dist = float(np.hypot(waypoint[0] - ax, waypoint[1] - ay))
        if dist < min_dist:
            min_dist = dist
            min_step = t + 1
    return min_dist, min_dist - collision_threshold, min_step


def _point_to_oriented_box_signed_distance(
    px: float,
    py: float,
    cx: float,
    cy: float,
    heading: float,
    length: float,
    width: float,
    buffer: float = 0.0,
) -> float:
    dx = px - cx
    dy = py - cy
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    qx = abs(local_x) - (0.5 * max(length, 0.0) + buffer)
    qy = abs(local_y) - (0.5 * max(width, 0.0) + buffer)
    outside = float(np.hypot(max(qx, 0.0), max(qy, 0.0)))
    inside = min(max(qx, qy), 0.0)
    return outside + inside


def path_oriented_box_margin_features(waypoints, agent, ego_radius: float = 1.0):
    min_box_distance = float("inf")
    min_collision_margin = float("inf")
    min_step = -1
    for t, waypoint in enumerate(waypoints):
        dt = (t + 1) * 0.5
        ax = agent.state.x + agent.state.vx * dt
        ay = agent.state.y + agent.state.vy * dt
        heading = agent.state.heading + agent.state.yaw_rate * dt
        box_distance = _point_to_oriented_box_signed_distance(
            waypoint[0],
            waypoint[1],
            ax,
            ay,
            heading,
            agent.state.length,
            agent.state.width,
            buffer=0.0,
        )
        collision_margin = _point_to_oriented_box_signed_distance(
            waypoint[0],
            waypoint[1],
            ax,
            ay,
            heading,
            agent.state.length,
            agent.state.width,
            buffer=ego_radius,
        )
        if collision_margin < min_collision_margin:
            min_box_distance = box_distance
            min_collision_margin = collision_margin
            min_step = t + 1
    return min_box_distance, min_collision_margin, min_step


def build_cluster_records(
    mode,
    scenario_index,
    frame_index,
    primary_sender,
    primary_perception,
    guarded_perception,
    evidence,
    temporal_labels,
    restored_reason,
    ego_only,
    hybrid,
    waypoints,
    args,
):
    _, ego_stats, _ = eval_method(hybrid, waypoints, ego_only)
    kept_keys = {agent_key(agent) for agent in guarded_perception.agents if agent.source == "coop"}
    records = []
    for agent in primary_perception.agents:
        if agent.source != "coop":
            continue
        details = evidence.cluster_details(agent)
        _, single_stats, _ = eval_method(hybrid, waypoints, single_agent_perception(ego_only, agent))
        geom_delta = single_stats.get("n_geometric_threats", 0) - ego_stats.get("n_geometric_threats", 0)
        mod_delta = single_stats.get("modification_rate", 0.0) - ego_stats.get("modification_rate", 0.0)
        distance, ttc, closest_distance = kinematic_impact_features(ego_only.ego, agent)
        path_min_distance, path_collision_margin, path_risk_step = path_margin_features(waypoints, agent)
        path_box_distance, path_box_collision_margin, path_box_risk_step = path_oriented_box_margin_features(
            waypoints,
            agent,
            ego_radius=args.path_box_ego_radius,
        )
        key = agent_key(agent)
        removed = key not in kept_keys
        supported = evidence.is_supported(agent)
        temporal = temporal_labels.get(key, {"temporal_status": "unknown", "unsupported_age": 0})
        if removed:
            final_action = "object_quarantine"
        elif key in restored_reason:
            final_action = f"{restored_reason[key]}_{temporal['temporal_status']}"
        elif not supported:
            final_action = f"probation_keep_{temporal['temporal_status']}"
        else:
            final_action = "keep"
        records.append(
            {
                "mode": mode,
                "scenario_index": scenario_index,
                "frame_index": frame_index,
                "primary_sender": primary_sender,
                "object_id": str(agent.state.id),
                "is_fake": int(str(agent.state.id) == "fake_front"),
                "x": float(agent.state.x),
                "y": float(agent.state.y),
                "evidence_supported": int(supported),
                "temporal_status": temporal["temporal_status"],
                "unsupported_age": temporal["unsupported_age"],
                "support_count": details["support_count"],
                "supporting_sender_ids": ";".join(details["supporting_sender_ids"]),
                "trust_weighted_support": details["trust_weighted_support"],
                "position_cov_trace": details["position_cov_trace"],
                "offset_spread": details["offset_spread"],
                "distance": distance,
                "ttc_s": ttc,
                "closest_distance": closest_distance,
                "path_min_distance": path_min_distance,
                "path_collision_margin": path_collision_margin,
                "path_risk_step": path_risk_step,
                "path_box_distance": path_box_distance,
                "path_box_collision_margin": path_box_collision_margin,
                "path_box_risk_step": path_box_risk_step,
                "geom_delta": geom_delta,
                "mod_delta": mod_delta,
                "final_action": final_action,
            }
        )
    return records


def run_mode(args, mode, support_modes, loader, val_idx, hybrid):
    rng = np.random.default_rng(args.seed)
    methods = ["EgoOnly", "CleanCoop", "PrimaryRaw", "PrimaryTrustCalib", "MultiPeerObjectGuard"]
    metrics = init_metrics(methods)
    cluster_records = []

    for si in val_idx:
        temporal_tracker = UnsupportedTemporalTracker(max_dist=args.temporal_match_dist)
        trust_state = SenderTrustState(args.primary_trust)
        scenario = loader.scenarios[si]
        total_frames = len(scenario["frames"])
        if args.frame_window == "full":
            frame_indices = list(range(total_frames))
        elif args.frame_window == "accident":
            collision_frame = scenario.get("collision_frame", -1)
            if collision_frame > 0:
                start = max(0, collision_frame - args.accident_window_before)
                end = min(total_frames, collision_frame + args.accident_window_after + 1)
                frame_indices = list(range(start, end))
            else:
                n_frames = total_frames if args.max_frames_per_scenario <= 0 else min(total_frames, args.max_frames_per_scenario)
                frame_indices = list(range(n_frames))
        else:
            n_frames = total_frames
            if args.max_frames_per_scenario > 0:
                n_frames = min(n_frames, args.max_frames_per_scenario)
            frame_indices = list(range(n_frames))

        for fi in frame_indices:
            frame = loader.load_frame(si, fi)
            waypoints = simulate_codriving_waypoints(frame)
            ego_only = filter_visible(frame.perception)
            clean_coop = merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)

            primary_msg, support_msgs = build_peer_messages(frame, mode, support_modes, rng, args)
            primary_raw = merge_ego_visible_with_coop_invisible(frame.perception, primary_msg)
            primary_trust, primary_report = calibrate_coop_message(frame.perception, primary_msg, args)

            support_trust = []
            support_reports = []
            for support_msg in support_msgs:
                perception, support_report = calibrate_coop_message(frame.perception, support_msg, args)
                support_trust.append(perception)
                support_reports.append(support_report)

            support_sender_ids = [f"support_{i}_{support_modes[i]}" for i in range(len(support_modes))]
            evidence = MultiPeerEvidenceSupport(
                support_trust,
                peer_reports=support_reports,
                sender_ids=support_sender_ids,
                sender_trusts=args.support_trusts,
                min_support=args.min_peer_support,
                min_trust_support=args.min_trust_support,
                max_dist=args.peer_match_dist,
            )
            (
                real_support_sum,
                real_support_n,
                fake_support_sum,
                fake_support_n,
                real_trust_support_sum,
                fake_trust_support_sum,
                real_cov_sum,
                fake_cov_sum,
                real_offset_spread_sum,
                fake_offset_spread_sum,
            ) = support_summary(primary_trust, evidence)
            guarded, guarded_report, obj_removed = object_impact_guard_perception(
                ego_only,
                primary_trust,
                primary_report,
                hybrid,
                waypoints,
                evidence_tracker=evidence,
            )
            guarded_report = copy.deepcopy(guarded_report)
            fake_seen = count_fake(primary_trust)
            temporal_labels = temporal_tracker.update_and_label(primary_trust, evidence)
            trust_before = trust_state.trust if args.enable_trust_dynamics else args.primary_trust
            original_primary_trust = args.primary_trust
            args.primary_trust = trust_before
            guarded, n_restored, restored_reason = apply_high_trust_probation(
                primary_trust,
                guarded,
                temporal_labels,
                args,
                ego_only,
                hybrid,
                waypoints,
            )
            args.primary_trust = original_primary_trust
            if n_restored:
                obj_removed = max(0, obj_removed - n_restored)
                guarded_report.n_output_agents = len(guarded.agents)
                guarded_report.n_corrected_agents += n_restored
                if obj_removed == 0:
                    guarded_report.action = "probation_restore"

            guarded, n_box_removed = apply_box_margin_guard(
                guarded,
                evidence,
                args,
                waypoints,
            )
            if n_box_removed:
                obj_removed += n_box_removed
                guarded_report.n_output_agents = len(guarded.agents)
                guarded_report.action = "object_quarantine"

            guarded, n_smoothed_agents, smooth_residual_sum = apply_peer_consensus_smoothing(
                guarded,
                support_trust,
                support_sender_ids,
                args.support_trusts,
                args,
            )
            if n_smoothed_agents:
                guarded_report.n_corrected_agents += n_smoothed_agents
                guarded_report.n_output_agents = len(guarded.agents)

            (
                guarded,
                n_missing_recovered,
                n_missing_candidates,
                n_missing_gt,
                n_missing_recovery_tp,
                n_missing_recovery_fp,
                missing_records,
            ) = apply_missing_object_recovery(
                mode,
                si,
                fi,
                guarded,
                primary_trust,
                primary_report,
                clean_coop,
                support_trust,
                support_sender_ids,
                args.support_trusts,
                args,
                ego_only,
                hybrid,
                waypoints,
            )
            if n_missing_recovered:
                guarded_report.n_output_agents = len(guarded.agents)
            fake_removed = max(0, fake_seen - count_fake(guarded))
            real_removed = count_real_removed(primary_trust, guarded)
            frame_records = []
            if args.write_cluster_records or args.enable_trust_dynamics:
                frame_records = build_cluster_records(
                    mode,
                    si,
                    fi,
                    f"primary_{mode}",
                    primary_trust,
                    guarded,
                    evidence,
                    temporal_labels,
                    restored_reason,
                    ego_only,
                    hybrid,
                    waypoints,
                    args,
                )
                for record in frame_records:
                    record["primary_trust_before"] = trust_before
                    record["primary_residual_after"] = primary_report.offset.residual_after
                    record["primary_correctable_score"] = primary_report.offset.correctable_score
                supported_covs = [
                    float(record["position_cov_trace"])
                    for record in frame_records
                    if int(record["evidence_supported"])
                ]
                peer_disagreement = float(np.mean(supported_covs)) if supported_covs else 0.0
                offset_instability = float(primary_report.offset.residual_after)
                for record in frame_records:
                    record["frame_offset_instability"] = offset_instability
                    record["frame_peer_disagreement"] = peer_disagreement
                if args.enable_trust_dynamics:
                    trust_after = trust_state.update(
                        frame_records,
                        penalty=args.trust_penalty,
                        reward=args.trust_reward,
                        offset_instability=offset_instability,
                        peer_disagreement=peer_disagreement,
                        offset_penalty=args.trust_offset_penalty,
                        disagreement_penalty=args.trust_disagreement_penalty,
                        offset_thr=args.trust_offset_thr,
                        disagreement_thr=args.trust_disagreement_thr,
                    )
                else:
                    trust_after = trust_before
                for record in frame_records:
                    record["primary_trust_after"] = trust_after
                if args.write_cluster_records:
                    cluster_records.extend(frame_records)
                    for record in missing_records:
                        record["primary_trust_before"] = trust_before
                        record["primary_residual_after"] = primary_report.offset.residual_after
                        record["primary_correctable_score"] = primary_report.offset.correctable_score
                        record["frame_offset_instability"] = offset_instability
                        record["frame_peer_disagreement"] = peer_disagreement
                        record["primary_trust_after"] = trust_after
                    cluster_records.extend(missing_records)

            per_method = {
                "EgoOnly": (ego_only, None, 0, 0, 0, 0),
                "CleanCoop": (clean_coop, None, 0, 0, 0, 0),
                "PrimaryRaw": (primary_raw, None, 0, 0, 0, 0),
                "PrimaryTrustCalib": (primary_trust, primary_report, 0, 0, 0, 0),
                "MultiPeerObjectGuard": (
                    guarded,
                    guarded_report,
                    obj_removed,
                    fake_seen,
                    fake_removed,
                    real_removed,
                    real_support_sum,
                    real_support_n,
                    fake_support_sum,
                    fake_support_n,
                    real_trust_support_sum,
                    fake_trust_support_sum,
                    real_cov_sum,
                    fake_cov_sum,
                    real_offset_spread_sum,
                    fake_offset_spread_sum,
                    n_missing_candidates,
                    n_missing_recovered,
                    n_missing_gt,
                    n_missing_recovery_tp,
                    n_missing_recovery_fp,
                    n_smoothed_agents,
                    smooth_residual_sum,
                ),
            }
            for name, values in per_method.items():
                perception, report, removed, fake_n, fake_removed_n, real_removed_n, *support_values = values
                if support_values:
                    (
                        real_sup_sum,
                        real_sup_n,
                        fake_sup_sum,
                        fake_sup_n,
                        real_trust_sum,
                        fake_trust_sum,
                        real_cov_sum_value,
                        fake_cov_sum_value,
                        real_offset_sum,
                        fake_offset_sum,
                        missing_candidate_count,
                        missing_recovered_count,
                        missing_gt_count,
                        missing_recovery_tp_count,
                        missing_recovery_fp_count,
                        smooth_agent_count,
                        smooth_residual_value,
                    ) = support_values
                else:
                    real_sup_sum, real_sup_n, fake_sup_sum, fake_sup_n = 0.0, 0, 0.0, 0
                    real_trust_sum = fake_trust_sum = 0.0
                    real_cov_sum_value = fake_cov_sum_value = 0.0
                    real_offset_sum = fake_offset_sum = 0.0
                    missing_candidate_count = missing_recovered_count = 0
                    missing_gt_count = missing_recovery_tp_count = missing_recovery_fp_count = 0
                    smooth_agent_count = 0
                    smooth_residual_value = 0.0
                modified, stats, warned = eval_method(hybrid, waypoints, perception)
                update(
                    metrics,
                    name,
                    frame,
                    modified,
                    stats,
                    warned,
                    report=report,
                    obj_removed=removed,
                    fake_seen=fake_n,
                    fake_removed=fake_removed_n,
                    real_removed=real_removed_n,
                    real_support_sum=real_sup_sum,
                    real_support_n=real_sup_n,
                    fake_support_sum=fake_sup_sum,
                    fake_support_n=fake_sup_n,
                    real_trust_support_sum=real_trust_sum,
                    fake_trust_support_sum=fake_trust_sum,
                    real_cov_sum=real_cov_sum_value,
                    fake_cov_sum=fake_cov_sum_value,
                    real_offset_spread_sum=real_offset_sum,
                    fake_offset_spread_sum=fake_offset_sum,
                    missing_candidates=missing_candidate_count,
                    missing_recovered=missing_recovered_count,
                    missing_gt=missing_gt_count,
                    missing_recovery_tp=missing_recovery_tp_count,
                    missing_recovery_fp=missing_recovery_fp_count,
                    smooth_agents=smooth_agent_count,
                    smooth_residual=smooth_residual_value,
                )

    return rows_from_metrics(mode, support_modes, metrics), cluster_records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="clean,shift,shift_severe,fake_front")
    parser.add_argument("--support-modes", default="clean,shift")
    parser.add_argument("--max-scenarios", type=int, default=20)
    parser.add_argument("--max-frames-per-scenario", type=int, default=20)
    parser.add_argument("--frame-window", choices=("start", "full", "accident"), default="start")
    parser.add_argument("--accident-window-before", type=int, default=20)
    parser.add_argument("--accident-window-after", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shift-x", type=float, default=2.0)
    parser.add_argument("--shift-y", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=1.5)
    parser.add_argument("--drop-rate", type=float, default=0.7)
    parser.add_argument("--stale-delay-s", type=float, default=1.0)
    parser.add_argument("--min-peer-support", type=int, default=1)
    parser.add_argument("--min-trust-support", type=float, default=0.0)
    parser.add_argument("--peer-match-dist", type=float, default=2.5)
    parser.add_argument("--temporal-match-dist", type=float, default=5.0)
    parser.add_argument("--support-trusts", default=None)
    parser.add_argument("--write-cluster-records", action="store_true")
    parser.add_argument("--primary-trust", type=float, default=1.0)
    parser.add_argument("--enable-high-trust-probation", action="store_true")
    parser.add_argument("--probation-primary-trust-thr", type=float, default=0.8)
    parser.add_argument("--probation-max-age", type=int, default=2)
    parser.add_argument("--probation-max-geom-delta", type=float, default=2.0)
    parser.add_argument("--probation-max-mod-delta", type=float, default=0.2)
    parser.add_argument("--probation-allow-high-impact", action="store_true")
    parser.add_argument("--enable-trust-dynamics", action="store_true")
    parser.add_argument("--trust-penalty", type=float, default=0.2)
    parser.add_argument("--trust-reward", type=float, default=0.02)
    parser.add_argument("--trust-offset-penalty", type=float, default=0.0)
    parser.add_argument("--trust-disagreement-penalty", type=float, default=0.0)
    parser.add_argument("--trust-offset-thr", type=float, default=0.5)
    parser.add_argument("--trust-disagreement-thr", type=float, default=1.0)
    parser.add_argument("--enable-missing-recovery", action="store_true")
    parser.add_argument("--missing-min-peer-support", type=int, default=2)
    parser.add_argument("--missing-min-trust-support", type=float, default=1.0)
    parser.add_argument("--missing-match-dist", type=float, default=2.5)
    parser.add_argument("--missing-cluster-dist", type=float, default=2.5)
    parser.add_argument("--missing-max-size-diff", type=float, default=3.0)
    parser.add_argument("--missing-recovery-primary-actions", default="accept")
    parser.add_argument("--missing-recover-high-impact-only", action="store_true")
    parser.add_argument("--missing-impact-geom-thr", type=float, default=1.0)
    parser.add_argument("--missing-impact-mod-thr", type=float, default=0.1)
    parser.add_argument("--path-box-ego-radius", type=float, default=1.0)
    parser.add_argument("--enable-box-margin-guard", action="store_true")
    parser.add_argument("--box-margin-guard-thr", type=float, default=0.0)
    parser.add_argument("--enable-time-calib", action="store_true")
    parser.add_argument("--time-delay-grid", default="0.0,0.25,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--time-correctable-thr", type=float, default=0.45)
    parser.add_argument("--time-downweight-thr", type=float, default=0.25)
    parser.add_argument("--time-clean-residual-thr", type=float, default=0.50)
    parser.add_argument("--time-min-delay-s", type=float, default=0.10)
    parser.add_argument("--time-min-residual-gain", type=float, default=0.05)
    parser.add_argument("--enable-peer-consensus-smoothing", action="store_true")
    parser.add_argument("--smooth-min-peer-support", type=int, default=2)
    parser.add_argument("--smooth-min-trust-support", type=float, default=1.0)
    parser.add_argument("--smooth-match-dist", type=float, default=4.0)
    parser.add_argument("--smooth-max-size-diff", type=float, default=3.0)
    parser.add_argument("--smooth-max-cluster-cov", type=float, default=4.0)
    parser.add_argument("--smooth-min-primary-residual", type=float, default=0.25)
    parser.add_argument("--smooth-alpha", type=float, default=1.0)
    parser.add_argument("--disable-v1", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/raid/xuyifan/trusted_coop_perception/results/deepaccident_multipeer_pilot"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.time_delay_grid_values = parse_delay_grid(args.time_delay_grid)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    support_modes = [mode.strip() for mode in args.support_modes.split(",") if mode.strip()]
    if args.support_trusts:
        support_trusts = [float(v.strip()) for v in args.support_trusts.split(",") if v.strip()]
        if len(support_trusts) != len(support_modes):
            raise ValueError("--support-trusts must have the same length as --support-modes")
    else:
        support_trusts = [1.0] * len(support_modes)
    args.support_trusts = support_trusts

    hybrid = load_hybrid(use_v1=not args.disable_v1)
    loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=False)
    ckpt = torch.load(SAFECO_ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    val_idx = list(ckpt.get("val_scenario_idx", []))
    available_val_scenarios = len(val_idx)
    if args.max_scenarios > 0:
        val_idx = val_idx[: args.max_scenarios]
    if args.max_scenarios > available_val_scenarios:
        print(
            f"Warning: requested max_scenarios={args.max_scenarios}, "
            f"but checkpoint val_scenario_idx contains only {available_val_scenarios}. "
            f"Using {len(val_idx)} scenarios."
        )

    all_rows = []
    all_cluster_records = []
    for mode in modes:
        print(f"Running mode={mode}, support_modes={support_modes}")
        rows, cluster_records = run_mode(args, mode, support_modes, loader, val_idx, hybrid)
        for row in rows:
            row["actual_scenarios"] = len(val_idx)
            row["available_val_scenarios"] = available_val_scenarios
            row["requested_max_scenarios"] = args.max_scenarios
            row["max_frames_per_scenario"] = args.max_frames_per_scenario
            row["frame_window"] = args.frame_window
            row["enable_missing_recovery"] = bool(args.enable_missing_recovery)
            row["enable_box_margin_guard"] = bool(args.enable_box_margin_guard)
            row["enable_peer_consensus_smoothing"] = bool(args.enable_peer_consensus_smoothing)
            row["disable_v1"] = bool(args.disable_v1)
        all_rows.extend(rows)
        all_cluster_records.extend(cluster_records)

    csv_path = args.out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    meta = vars(args).copy()
    meta["modes"] = modes
    meta["support_modes"] = support_modes
    meta["available_val_scenarios"] = available_val_scenarios
    meta["val_scenarios_used"] = len(val_idx)
    meta["val_scenario_idx_used"] = val_idx
    meta["disable_v1"] = bool(args.disable_v1)
    meta["out_dir"] = str(args.out_dir)
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if all_cluster_records:
        record_path = args.out_dir / "cluster_records.csv"
        record_fieldnames = []
        for record in all_cluster_records:
            for key in record.keys():
                if key not in record_fieldnames:
                    record_fieldnames.append(key)
        with record_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record_fieldnames)
            writer.writeheader()
            writer.writerows(all_cluster_records)
    else:
        record_path = None

    print(f"\n{'mode':13s} {'method':22s} {'WPC%':>7s} {'warn%':>7s} {'p':>6s} {'corr':>6s} {'objQ':>6s} {'prob':>6s} {'smooth':>8s} {'miss':>8s} {'rm/fr':>8s} {'fake':>8s} {'real/fr':>8s} {'sup R/F':>9s} {'wt R/F':>9s} {'cov R/F':>11s}")
    for row in all_rows:
        if row["method"] not in ("CleanCoop", "PrimaryRaw", "PrimaryTrustCalib", "MultiPeerObjectGuard"):
            continue
        print(
            f"{row['mode']:13s} {row['method']:22s} {100*row['WPC']:6.2f}% "
            f"{100*row['warn_rate']:6.2f}% {row['avg_p_coll']:6.3f} {100*row['correct_rate']:5.1f}% "
            f"{100*row['object_quarantine_rate']:5.1f}% {100*row['probation_restore_rate']:5.1f}% "
            f"{row['avg_smoothed_agents']:4.2f}/fr "
            f"{row['avg_missing_recovered']:4.2f}/fr "
            f"{row['avg_obj_removed']:4.2f}/fr "
            f"{100*row['fake_removal_rate']:6.1f}% {row['clean_false_remove_per_frame']:7.3f} "
            f"{row['avg_real_peer_support']:4.1f}/{row['avg_fake_peer_support']:3.1f} "
            f"{row['avg_real_trust_support']:4.1f}/{row['avg_fake_trust_support']:3.1f} "
            f"{row['avg_real_cluster_cov']:5.2f}/{row['avg_fake_cluster_cov']:4.2f}"
        )
    print(f"\nWrote {csv_path}")
    if record_path is not None:
        print(f"Wrote {record_path}")


if __name__ == "__main__":
    main()

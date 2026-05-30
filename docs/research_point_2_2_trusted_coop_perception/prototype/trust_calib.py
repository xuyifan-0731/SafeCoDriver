"""Minimal TrustCalib prototype for DeepAccident object-level messages.

This prototype is deliberately small and object-level. It does not modify the
SafeCoDriver codebase. The goal is to test the first research claim:
stable spatially shifted cooperative information should be corrected rather
than blindly fused or discarded.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np

from coop_safety.interface import Agent, PerceptionResult


@dataclass
class OffsetEstimate:
    dx: float = 0.0
    dy: float = 0.0
    delay_s: float = 0.0
    residual_before: float = 0.0
    residual_after: float = 0.0
    inlier_ratio: float = 0.0
    match_count: int = 0
    correctable_score: float = 0.0


@dataclass
class CalibReport:
    action: str
    message_usability: float
    offset: OffsetEstimate
    n_input_coop_agents: int
    n_output_agents: int
    n_corrected_agents: int


def copy_agent(agent: Agent) -> Agent:
    return copy.deepcopy(agent)


def filter_visible(perception: PerceptionResult) -> PerceptionResult:
    return PerceptionResult(
        timestamp=perception.timestamp,
        ego=copy.deepcopy(perception.ego),
        agents=[copy_agent(a) for a in perception.agents if a.is_visible],
        lanes=copy.deepcopy(perception.lanes),
        blind_spots=copy.deepcopy(perception.blind_spots),
        visibility_map=copy.deepcopy(perception.visibility_map),
    )


def shifted_message(perception: PerceptionResult, dx: float, dy: float) -> PerceptionResult:
    msg = copy.deepcopy(perception)
    for agent in msg.agents:
        agent.state.x += dx
        agent.state.y += dy
        agent.source = "coop_shifted"
    return msg


def oracle_correct_message(message: PerceptionResult, dx: float, dy: float) -> PerceptionResult:
    msg = copy.deepcopy(message)
    for agent in msg.agents:
        agent.state.x -= dx
        agent.state.y -= dy
        agent.source = "coop_oracle"
    return msg


def _agent_distance(a: Agent, b: Agent) -> float:
    return math.hypot(a.state.x - b.state.x, a.state.y - b.state.y)


def _size_distance(a: Agent, b: Agent) -> float:
    return abs(a.state.length - b.state.length) + abs(a.state.width - b.state.width)


def greedy_associate(
    ego_agents: list[Agent],
    coop_agents: list[Agent],
    max_dist: float = 8.0,
    max_size_diff: float = 3.0,
) -> list[tuple[Agent, Agent, float]]:
    """Greedy association without using IDs.

    This keeps the pilot closer to a real object-list setting than matching by
    DeepAccident object id.
    """
    candidates = []
    for i, ea in enumerate(ego_agents):
        for j, ca in enumerate(coop_agents):
            if ea.agent_type != ca.agent_type:
                continue
            dist = _agent_distance(ea, ca)
            if dist > max_dist:
                continue
            size = _size_distance(ea, ca)
            if size > max_size_diff:
                continue
            cost = dist + 0.5 * size
            candidates.append((cost, i, j))

    candidates.sort()
    used_e: set[int] = set()
    used_c: set[int] = set()
    matches: list[tuple[Agent, Agent, float]] = []
    for cost, i, j in candidates:
        if i in used_e or j in used_c:
            continue
        used_e.add(i)
        used_c.add(j)
        matches.append((ego_agents[i], coop_agents[j], cost))
    return matches


def estimate_translation_offset(matches: list[tuple[Agent, Agent, float]]) -> OffsetEstimate:
    if not matches:
        return OffsetEstimate()

    deltas = np.array(
        [[ea.state.x - ca.state.x, ea.state.y - ca.state.y] for ea, ca, _ in matches],
        dtype=float,
    )
    residual_before = float(np.mean(np.linalg.norm(deltas, axis=1)))

    med = np.median(deltas, axis=0)
    dist_to_med = np.linalg.norm(deltas - med, axis=1)
    med_abs = float(np.median(np.abs(dist_to_med - np.median(dist_to_med))))
    gate = max(1.5, 3.0 * med_abs)
    inliers = dist_to_med <= gate
    if not np.any(inliers):
        inliers = np.ones(len(deltas), dtype=bool)

    offset = np.mean(deltas[inliers], axis=0)
    corrected = deltas - offset
    residual_after_all = np.linalg.norm(corrected, axis=1)
    residual_after = float(np.mean(residual_after_all[inliers]))
    inlier_ratio = float(np.mean(inliers))

    improvement = 0.0
    if residual_before > 1e-6:
        improvement = max(0.0, (residual_before - residual_after) / residual_before)
    count_score = min(len(matches) / 4.0, 1.0)
    residual_score = math.exp(-residual_after / 2.0)
    correctable = float(np.clip(0.50 * improvement + 0.25 * inlier_ratio + 0.25 * count_score, 0.0, 1.0))
    correctable *= residual_score

    return OffsetEstimate(
        dx=float(offset[0]),
        dy=float(offset[1]),
        residual_before=residual_before,
        residual_after=residual_after,
        inlier_ratio=inlier_ratio,
        match_count=len(matches),
        correctable_score=correctable,
    )


def merge_ego_visible_with_coop_invisible(
    ego_perception: PerceptionResult,
    coop_message: PerceptionResult,
    confidence_scale: float = 1.0,
) -> PerceptionResult:
    """Use ego direct perception plus cooperative-only blind-spot agents."""
    agents: list[Agent] = []
    for agent in ego_perception.agents:
        if agent.is_visible:
            a = copy_agent(agent)
            a.source = "ego"
            a.confidence = 1.0
            agents.append(a)

    for agent in coop_message.agents:
        if not agent.is_visible:
            a = copy_agent(agent)
            a.source = "coop"
            a.confidence = max(0.0, min(1.0, a.confidence * confidence_scale))
            agents.append(a)

    return PerceptionResult(
        timestamp=ego_perception.timestamp,
        ego=copy.deepcopy(ego_perception.ego),
        agents=agents,
        lanes=copy.deepcopy(ego_perception.lanes),
        blind_spots=copy.deepcopy(ego_perception.blind_spots),
        visibility_map=copy.deepcopy(ego_perception.visibility_map),
    )


def calibrate_shifted_coop(
    ego_perception: PerceptionResult,
    coop_message: PerceptionResult,
    correctable_thr: float = 0.55,
    downweight_thr: float = 0.30,
    clean_residual_thr: float = 0.50,
) -> tuple[PerceptionResult, CalibReport]:
    ego_visible = [a for a in ego_perception.agents if a.is_visible]
    coop_visible = [a for a in coop_message.agents if a.is_visible]
    matches = greedy_associate(ego_visible, coop_visible)
    offset = estimate_translation_offset(matches)

    corrected = copy.deepcopy(coop_message)
    for agent in corrected.agents:
        agent.state.x += offset.dx
        agent.state.y += offset.dy

    if offset.match_count >= 3 and offset.residual_before <= clean_residual_thr:
        action = "accept"
        usability = 1.0
        output = merge_ego_visible_with_coop_invisible(ego_perception, coop_message, usability)
        n_corrected = 0
    elif offset.correctable_score >= correctable_thr:
        action = "correct"
        usability = min(1.0, 0.55 + 0.45 * offset.correctable_score)
        output = merge_ego_visible_with_coop_invisible(ego_perception, corrected, usability)
        n_corrected = sum(1 for a in corrected.agents if not a.is_visible)
    elif offset.correctable_score >= downweight_thr:
        action = "downweight"
        usability = offset.correctable_score
        output = merge_ego_visible_with_coop_invisible(ego_perception, corrected, usability)
        n_corrected = sum(1 for a in corrected.agents if not a.is_visible)
    else:
        action = "quarantine"
        usability = 0.0
        output = filter_visible(ego_perception)
        n_corrected = 0

    return output, CalibReport(
        action=action,
        message_usability=usability,
        offset=offset,
        n_input_coop_agents=len(coop_message.agents),
        n_output_agents=len(output.agents),
        n_corrected_agents=n_corrected,
    )


def _propagate_message(message: PerceptionResult, delay_s: float) -> PerceptionResult:
    msg = copy.deepcopy(message)
    for agent in msg.agents:
        agent.state.x += agent.state.vx * delay_s
        agent.state.y += agent.state.vy * delay_s
        agent.state.heading += agent.state.yaw_rate * delay_s
    return msg


def estimate_temporal_delay(
    ego_visible: list[Agent],
    coop_visible: list[Agent],
    delay_grid: list[float],
) -> OffsetEstimate:
    if not ego_visible or not coop_visible:
        return OffsetEstimate()

    baseline_matches = greedy_associate(ego_visible, coop_visible)
    if not baseline_matches:
        return OffsetEstimate()
    residual_before = float(np.mean([_agent_distance(ea, ca) for ea, ca, _ in baseline_matches]))

    best_delay = 0.0
    best_residual = residual_before
    best_match_count = len(baseline_matches)
    for delay_s in delay_grid:
        propagated = []
        for agent in coop_visible:
            a = copy_agent(agent)
            a.state.x += a.state.vx * delay_s
            a.state.y += a.state.vy * delay_s
            a.state.heading += a.state.yaw_rate * delay_s
            propagated.append(a)
        matches = greedy_associate(ego_visible, propagated)
        if not matches:
            continue
        residual = float(np.mean([_agent_distance(ea, ca) for ea, ca, _ in matches]))
        if residual < best_residual:
            best_delay = float(delay_s)
            best_residual = residual
            best_match_count = len(matches)

    improvement = 0.0
    if residual_before > 1e-6:
        improvement = max(0.0, (residual_before - best_residual) / residual_before)
    count_score = min(best_match_count / 4.0, 1.0)
    residual_score = math.exp(-best_residual / 2.0)
    correctable = float(np.clip(0.60 * improvement + 0.20 * count_score + 0.20 * min(best_delay / 1.0, 1.0), 0.0, 1.0))
    correctable *= residual_score

    return OffsetEstimate(
        dx=0.0,
        dy=0.0,
        delay_s=best_delay,
        residual_before=residual_before,
        residual_after=best_residual,
        inlier_ratio=1.0,
        match_count=best_match_count,
        correctable_score=correctable,
    )


def calibrate_temporal_coop(
    ego_perception: PerceptionResult,
    coop_message: PerceptionResult,
    delay_grid: list[float],
    correctable_thr: float = 0.45,
    downweight_thr: float = 0.25,
    clean_residual_thr: float = 0.50,
    min_delay_s: float = 0.10,
) -> tuple[PerceptionResult, CalibReport]:
    ego_visible = [a for a in ego_perception.agents if a.is_visible]
    coop_visible = [a for a in coop_message.agents if a.is_visible]
    offset = estimate_temporal_delay(ego_visible, coop_visible, delay_grid)

    corrected = _propagate_message(coop_message, offset.delay_s)
    if offset.match_count >= 3 and offset.residual_before <= clean_residual_thr:
        action = "accept"
        usability = 1.0
        output = merge_ego_visible_with_coop_invisible(ego_perception, coop_message, usability)
        n_corrected = 0
    elif offset.delay_s >= min_delay_s and offset.correctable_score >= correctable_thr:
        action = "time_correct"
        usability = min(1.0, 0.55 + 0.45 * offset.correctable_score)
        output = merge_ego_visible_with_coop_invisible(ego_perception, corrected, usability)
        n_corrected = sum(1 for a in corrected.agents if not a.is_visible)
    elif offset.delay_s >= min_delay_s and offset.correctable_score >= downweight_thr:
        action = "downweight"
        usability = offset.correctable_score
        output = merge_ego_visible_with_coop_invisible(ego_perception, corrected, usability)
        n_corrected = sum(1 for a in corrected.agents if not a.is_visible)
    else:
        action = "quarantine"
        usability = 0.0
        output = filter_visible(ego_perception)
        n_corrected = 0

    return output, CalibReport(
        action=action,
        message_usability=usability,
        offset=offset,
        n_input_coop_agents=len(coop_message.agents),
        n_output_agents=len(output.agents),
        n_corrected_agents=n_corrected,
    )

"""Shared method variants for final DeepAccident and SUMO evaluation."""
from __future__ import annotations

import math
import numpy as np

from coop_safety.interface import PerceptionResult


class HybridV1Only:
    """Ablation: V1 detector only, no waypoint correction."""

    name = "Hybrid-V1Only"

    def __init__(self, base, brake_factor: float = 0.7):
        self.base = base
        self.brake_factor = brake_factor

    def constrain_waypoints(self, waypoints, perception):
        prob = self.base._detect_with_v1(perception)
        fire = prob > self.base.detection_threshold
        return waypoints.copy(), {
            "method": self.name,
            "n_collisions_detected": 1 if fire else 0,
            "n_geometric_threats": 0,
            "modification_rate": 0.0,
            "collision_prob": prob,
            "target_speed_factor": self.brake_factor if fire else 1.0,
            "min_ttc": self.base._compute_min_ttc(perception),
            "mode": "V1_ONLY" if fire else "NORMAL",
        }


class HybridGeometryOnly:
    """Ablation: geometric waypoint correction only, no V1 detection."""

    name = "Hybrid-GeomOnly"

    def __init__(self, base):
        self.base = base

    def constrain_waypoints(self, waypoints, perception):
        old_detector = self.base.detector
        old_threshold = self.base.detection_threshold
        try:
            self.base.detector = None
            self.base.detection_threshold = 1.1
            mw, stats = self.base.constrain_waypoints(waypoints, perception)
        finally:
            self.base.detector = old_detector
            self.base.detection_threshold = old_threshold
        stats["method"] = self.name
        stats["collision_prob"] = 0.0
        stats["n_collisions_detected"] = 1 if stats.get("n_geometric_threats", 0) > 0 else 0
        return mw, stats


class HybridGeometryTTC:
    """Ablation: geometric correction plus TTC trigger, no V1 detection."""

    name = "Hybrid-Geom+TTC"

    def __init__(self, base, ttc_override: float = 3.0):
        self.base = base
        self.ttc_override = ttc_override

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = HybridGeometryOnly(self.base).constrain_waypoints(waypoints, perception)
        min_ttc = stats.get("min_ttc", 999.0)
        n_geom = stats.get("n_geometric_threats", 0)
        fire = n_geom > 0 or min_ttc < self.ttc_override
        stats["method"] = self.name
        stats["n_collisions_detected"] = 1 if fire else 0
        if min_ttc < 2.0:
            stats["target_speed_factor"] = min(stats.get("target_speed_factor", 1.0), 0.1)
        elif min_ttc < self.ttc_override:
            stats["target_speed_factor"] = min(stats.get("target_speed_factor", 1.0), 0.3)
        return mw, stats


class HybridWithGeometricANDTTC:
    """AND-fusion with an imminent TTC override."""

    name = "Ours-Hybrid+AND+TTC"

    def __init__(self, base, ttc_override: float = 3.0):
        self.base = base
        self.ttc_override = ttc_override

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get("collision_prob", 0.0)
        n_geom = stats.get("n_geometric_threats", 0)
        min_ttc = stats.get("min_ttc", 999.0)
        fire = ((prob > self.base.detection_threshold) and (n_geom > 0)) or (min_ttc < self.ttc_override)
        stats["n_collisions_detected"] = 1 if fire else 0
        if min_ttc < 2.0:
            stats["target_speed_factor"] = min(stats.get("target_speed_factor", 1.0), 0.1)
        elif min_ttc < self.ttc_override:
            stats["target_speed_factor"] = min(stats.get("target_speed_factor", 1.0), 0.3)
        return mw, stats


def estimate_directional_ttc(perception: PerceptionResult | None) -> tuple[float, float]:
    """Return (front_or_side_ttc, rear_ttc) in the ego frame."""
    if perception is None:
        return 999.0, 999.0
    front_side_ttc = 999.0
    rear_ttc = 999.0
    for agent in perception.agents:
        s = agent.state
        dist = math.hypot(s.x, s.y)
        if dist < 0.5:
            ttc = 0.0
        else:
            approach = -(s.x * s.vx + s.y * s.vy) / max(dist, 1e-6)
            if approach <= 0.1:
                continue
            ttc = dist / approach
        if s.x < -1.0 and abs(s.y) < 3.5:
            rear_ttc = min(rear_ttc, ttc)
        else:
            front_side_ttc = min(front_side_ttc, ttc)
    return front_side_ttc, rear_ttc


def estimate_rear_gap(perception: PerceptionResult | None) -> float:
    """Nearest same-lane rear vehicle distance in ego coordinates."""
    if perception is None:
        return 999.0
    rear_gap = 999.0
    for agent in perception.agents:
        s = agent.state
        if s.x < -1.0 and abs(s.y) < 3.5:
            rear_gap = min(rear_gap, abs(s.x))
    return rear_gap


class HybridWithGeometricANDTTCAware:
    """AND+TTC variant that avoids converting rear risk into hard braking."""

    name = "Ours-Hybrid+AND+TTC+RearAware"

    def __init__(self, base, ttc_override: float = 3.0):
        self.base = base
        self.ttc_override = ttc_override

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get("collision_prob", 0.0)
        n_geom = stats.get("n_geometric_threats", 0)
        front_ttc, rear_ttc = estimate_directional_ttc(perception)

        fire = ((prob > self.base.detection_threshold) and (n_geom > 0)) or (front_ttc < self.ttc_override)
        stats["n_collisions_detected"] = 1 if fire else 0
        stats["front_side_ttc"] = front_ttc
        stats["rear_ttc"] = rear_ttc

        if front_ttc < 2.0:
            factor = 0.1
        elif front_ttc < self.ttc_override:
            factor = 0.3
        elif fire:
            factor = min(stats.get("target_speed_factor", 1.0), 0.7)
        else:
            factor = 1.0

        if rear_ttc < 2.5 and rear_ttc < front_ttc:
            factor = max(factor, 0.7)
        stats["target_speed_factor"] = factor
        return mw, stats

class HybridWithGeometricANDTTCMinHarm:
    """Prefer the lower-severity impact when an aggressive rear car is close."""

    name = "Ours-Hybrid+AND+TTC+MinHarm"

    def __init__(self, base, ttc_override: float = 3.0, rear_gap_guard: float = 18.0):
        self.base = base
        self.ttc_override = ttc_override
        self.rear_gap_guard = rear_gap_guard

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get("collision_prob", 0.0)
        n_geom = stats.get("n_geometric_threats", 0)
        front_ttc, rear_ttc = estimate_directional_ttc(perception)
        rear_gap = estimate_rear_gap(perception)

        fire = ((prob > self.base.detection_threshold) and (n_geom > 0)) or (front_ttc < self.ttc_override)
        stats["n_collisions_detected"] = 1 if fire else 0
        stats["front_side_ttc"] = front_ttc
        stats["rear_ttc"] = rear_ttc
        stats["rear_gap"] = rear_gap

        if front_ttc < 1.0:
            factor = 0.3
        elif front_ttc < self.ttc_override:
            factor = 0.7
        elif fire:
            factor = 0.8
        else:
            factor = 1.0

        if rear_gap < self.rear_gap_guard and front_ttc >= 1.5:
            factor = max(factor, 1.0)
        elif rear_ttc < 2.5 and rear_ttc < front_ttc:
            factor = max(factor, 0.8)
        stats["target_speed_factor"] = factor
        return mw, stats


class HybridWithGeometricANDTTCRearEscape:
    """Avoid front/side conflict while escaping close aggressive rear traffic."""

    name = "Ours-Hybrid+AND+TTC+RearEscape"

    def __init__(self, base, ttc_override: float = 3.0, rear_gap_guard: float = 18.0):
        self.base = base
        self.ttc_override = ttc_override
        self.rear_gap_guard = rear_gap_guard

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get("collision_prob", 0.0)
        n_geom = stats.get("n_geometric_threats", 0)
        front_ttc, rear_ttc = estimate_directional_ttc(perception)
        rear_gap = estimate_rear_gap(perception)

        fire = ((prob > self.base.detection_threshold) and (n_geom > 0)) or (front_ttc < self.ttc_override)
        stats["n_collisions_detected"] = 1 if fire else 0
        stats["front_side_ttc"] = front_ttc
        stats["rear_ttc"] = rear_ttc
        stats["rear_gap"] = rear_gap

        close_rear = rear_gap < self.rear_gap_guard or rear_ttc < 2.5
        if close_rear and front_ttc >= 1.0:
            stats["lane_escape"] = 1
            factor = 1.0
        elif front_ttc < 1.0:
            stats["lane_escape"] = 1 if close_rear else 0
            factor = 0.5
        elif front_ttc < self.ttc_override:
            factor = 0.7
        elif fire:
            factor = 0.8
        else:
            factor = 1.0

        stats["target_speed_factor"] = factor
        return mw, stats

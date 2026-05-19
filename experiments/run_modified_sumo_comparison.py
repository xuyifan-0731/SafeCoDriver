"""Compare methods on the revised 10 SUMO blind-spot scenarios.

This runner follows the latest SUMO metrics in
docs/260511_盲区场景统一评测.md, but uses the revised scenario
definitions from experiments/gen_scenario_gifs.py:

  BS1..BS5 x blind/nonblind

Metrics:
  CollRate, secondary collisions, severity, Det(s), Det(f), Early,
  FA(s), FA(f), WPC%, Mod%.
"""
from __future__ import annotations

import csv
import argparse
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

os.environ["SUMO_HOME"] = str(Path(sys.executable).parent.parent / "lib/python3.10/site-packages/sumo")
import traci

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from coop_safety.interface import Agent, AgentType, ConstraintMode, PerceptionResult, VehicleState
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments import gen_scenario_gifs as scene_gen
from experiments.methods import RSSOnly
from experiments.methods_new_baselines import MAPSafety, RiskMMSafety, UniE2EV2XSafety
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND


OUT_DIR = ROOT / "results" / "modified_sumo"
EGO_RANGE = 50.0
COOP_RANGE = 50.0
BLOCKER_RADIUS = 4.0


@dataclass
class RunMetrics:
    n_frames: int = 0
    unique_ego_coll: int = 0
    secondary_coll: int = 0
    severities: list[float] = field(default_factory=list)
    first_collision: int = -1
    first_warning: int = -1
    n_warning_frames: int = 0
    n_dangerous_frames: int = 0
    n_warned_dangerous: int = 0
    n_safe_frames: int = 0
    n_warned_safe: int = 0
    wp_coll: int = 0
    wp_total: int = 0
    n_modified_frames: int = 0
    overlap_coll: int = 0
    first_overlap: int = -1
    collision_pairs: list[str] = field(default_factory=list)


@dataclass
class Agg:
    n_runs: int = 0
    coll_runs: int = 0
    ego_coll_total: int = 0
    secondary_total: int = 0
    severities: list[float] = field(default_factory=list)
    n_frames: int = 0
    n_dangerous_frames: int = 0
    n_warned_dangerous: int = 0
    n_safe_frames: int = 0
    n_warned_safe: int = 0
    n_collision_truth: int = 0
    n_detect_scen: int = 0
    n_normal_truth: int = 0
    n_fa_scen: int = 0
    early_warnings: list[int] = field(default_factory=list)
    wp_coll: int = 0
    wp_total: int = 0
    n_modified_frames: int = 0
    overlap_runs: int = 0
    overlap_total: int = 0


class HybridWithGeometricANDTTC:
    """AND-fusion with an imminent-risk override.

    Normal AND: V1 must agree with geometric threats to reduce false alarms.
    Override: if TTC is already short, trigger even when V1 lags.
    """
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

        # If a close rear vehicle is not yet closing, hard braking creates the
        # collision. In that case, keep ego moving unless the front/side threat
        # is already immediate.
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
            # Hard braking created the rear impacts in stress BS3/BS4. Prefer a
            # lateral escape and keep speed unless the front threat is immediate.
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


def get_vehicle(vid: str):
    try:
        x, y = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        angle = traci.vehicle.getAngle(vid)
        heading = math.radians(90 - angle)
        return x, y, speed, heading, traci.vehicle.getLength(vid), traci.vehicle.getWidth(vid)
    except Exception:
        return None


def is_los_blocked(ego_pos, target_pos, blockers_pos, radius=BLOCKER_RADIUS) -> bool:
    ex, ey = ego_pos
    tx, ty = target_pos
    sx, sy = tx - ex, ty - ey
    seg_len2 = sx * sx + sy * sy
    if seg_len2 < 1e-6:
        return False
    for bx, by in blockers_pos:
        t = ((bx - ex) * sx + (by - ey) * sy) / seg_len2
        if t < 0.0 or t > 1.0:
            continue
        px, py = ex + t * sx, ey + t * sy
        if math.hypot(bx - px, by - py) < radius:
            return True
    return False


def build_perception(ego_uses_coop: bool) -> PerceptionResult | None:
    vids = list(traci.vehicle.getIDList())
    ego = get_vehicle("ego")
    if ego is None:
        return None
    ex, ey, esp, eh, el, ew = ego
    evx, evy = esp * math.cos(eh), esp * math.sin(eh)

    coop_pos = None
    if ego_uses_coop and "coop" in vids:
        coop = get_vehicle("coop")
        if coop:
            coop_pos = (coop[0], coop[1])

    blockers = []
    for vid in vids:
        if vid.startswith("blk_"):
            b = get_vehicle(vid)
            if b:
                blockers.append((b[0], b[1]))

    agents = []
    for vid in vids:
        if vid == "ego" or (vid == "coop" and ego_uses_coop):
            continue
        data = get_vehicle(vid)
        if data is None:
            continue
        x, y, sp, h, length, width = data
        d_ego = math.hypot(x - ex, y - ey)
        visible_ego = d_ego <= EGO_RANGE and not is_los_blocked((ex, ey), (x, y), blockers)
        visible_coop = False
        if coop_pos:
            d_coop = math.hypot(x - coop_pos[0], y - coop_pos[1])
            visible_coop = d_coop <= COOP_RANGE and not is_los_blocked(coop_pos, (x, y), blockers)
        if not visible_ego and not visible_coop:
            continue

        ch, sh = math.cos(-eh), math.sin(-eh)
        dx, dy = x - ex, y - ey
        rx, ry = dx * ch - dy * sh, dx * sh + dy * ch
        vx, vy = sp * math.cos(h), sp * math.sin(h)
        rvx = (vx - evx) * ch - (vy - evy) * sh
        rvy = (vx - evx) * sh + (vy - evy) * ch
        vtype = "pedestrian" if traci.vehicle.getTypeID(vid) == "pedestrian" else "car"
        agents.append(Agent(
            state=VehicleState(
                id=vid, x=rx, y=ry, heading=h - eh, velocity=sp,
                vx=rvx, vy=rvy, length=length, width=width,
                vehicle_type=vtype,
            ),
            agent_type=AgentType.PEDESTRIAN if vtype == "pedestrian" else AgentType.VEHICLE,
            is_visible=visible_ego,
            confidence=1.0 if visible_ego else 0.7,
        ))

    return PerceptionResult(
        timestamp=traci.simulation.getTime(),
        ego=VehicleState(id="ego", x=0.0, y=0.0, heading=0.0,
                         velocity=esp, vx=esp, vy=0.0, length=el, width=ew),
        agents=agents,
    )


def estimate_min_ttc(perception: PerceptionResult | None) -> float:
    if perception is None:
        return 999.0
    min_ttc = 999.0
    for agent in perception.agents:
        s = agent.state
        dist = math.hypot(s.x, s.y)
        if dist < 0.5:
            return 0.0
        approach = -(s.x * s.vx + s.y * s.vy) / max(dist, 1e-6)
        if approach > 0.1:
            min_ttc = min(min_ttc, dist / approach)
    return min_ttc


def apply_scene_controls(scenario_name: str, step: int, vids: list[str]) -> None:
    """Replay the revised scenario controls while leaving ego to the method."""
    for vid in vids:
        try:
            traci.vehicle.setLaneChangeMode(vid, 0)
        except Exception:
            pass
        if vid.startswith("attacker"):
            try:
                traci.vehicle.setSpeedMode(vid, 0)
            except Exception:
                pass
        if vid == "attacker_rear":
            try:
                traci.vehicle.setSpeed(vid, 13.5)
            except Exception:
                pass

        if scenario_name.startswith("BS1") and vid == "attacker":
            try:
                t = step * 0.1
                if 3.9 <= t <= 5.2:
                    if t <= 4.6:
                        phase = (t - 3.9) / 0.7
                        x = 72.2 - 1.8 * phase
                        y = 61.6 - 3.2 * phase
                        angle = 220 - 120 * phase
                    else:
                        phase = (t - 4.6) / 0.6
                        x = 70.4 + 7.0 * phase
                        y = 58.4 - 3.2 * phase
                        angle = 100
                    traci.vehicle.moveToXY(vid, "", -1, x, y, angle=angle, keepRoute=2)
                if traci.vehicle.getLaneID(vid).startswith("B1C1"):
                    traci.vehicle.changeLane(vid, 0, 3.0)
                    traci.vehicle.setSpeed(vid, 8.5)
            except Exception:
                pass

        if scenario_name.startswith("BS2") and vid == "attacker":
            try:
                t = step * 0.1
                x = 91.0
                y = -9.0 + 3.6 * max(0.0, t - 0.7)
                y = min(y, 3.5)
                traci.vehicle.moveToXY(vid, "", -1, x, y, angle=0, keepRoute=2)
                traci.vehicle.setSpeed(vid, 0)
            except Exception:
                pass

        if scenario_name.startswith("BS3") and vid in ("attacker", "blk_parallel"):
            try:
                if vid == "attacker":
                    traci.vehicle.setSpeed(vid, 11.5)
                    t = step * 0.1
                    # Force the corner attacker through the ego lane at the
                    # conflict point so SUMO reports a true collision instead
                    # of only a visual overlap inside the junction.
                    if 4.4 <= t <= 5.0:
                        phase = (t - 4.4) / 0.6
                        traci.vehicle.moveToXY(
                            vid, "", -1,
                            404.2 + 3.6 * phase,
                            395.2,
                            angle=90,
                            keepRoute=2,
                        )
                else:
                    _, y = traci.vehicle.getPosition(vid)
                    if y < 378.0:
                        traci.vehicle.setSpeed(vid, 11.5)
                    else:
                        traci.vehicle.slowDown(vid, 0, 0.7)
            except Exception:
                pass

        if scenario_name.startswith("BS4") and vid == "attacker":
            try:
                if traci.vehicle.getLaneID(vid).startswith("A1B1"):
                    traci.vehicle.changeLane(vid, 1, 3.0)
                    traci.vehicle.setSpeed(vid, 11.5)
            except Exception:
                pass
        if scenario_name.startswith("BS4") and vid == "blk_right":
            try:
                x, _ = traci.vehicle.getPosition(vid)
                if x < 386.0:
                    traci.vehicle.setSpeed(vid, 11.5)
                else:
                    traci.vehicle.slowDown(vid, 0, 0.8)
            except Exception:
                pass

        if scenario_name.startswith("BS5") and vid == "attacker" and step >= 20:
            try:
                traci.vehicle.setSpeed(vid, max(0, 11 - (step - 20) * 1.5))
            except Exception:
                pass
        if scenario_name.startswith("BS5") and vid == "midcar" and step >= 30:
            try:
                if traci.vehicle.getLaneIndex(vid) > 0:
                    traci.vehicle.changeLane(vid, 0, 3.0)
            except Exception:
                pass


def current_frame(vids: list[str]) -> dict[str, tuple]:
    out = {}
    for vid in vids:
        data = get_vehicle(vid)
        if data is None:
            continue
        x, y, sp, h, _, _ = data
        out[vid] = (x, y, h, traci.vehicle.getTypeID(vid), sp)
    return out


def visual_collision_pairs(frame: dict[str, tuple], seen: set[tuple[str, str]]) -> list[tuple[str, str]]:
    pairs = []
    if "ego" not in frame:
        return pairs
    for vid in ("attacker", "attacker_rear", "blk_bus", "blk_park", "blk_parallel", "blk_right", "midcar"):
        if vid not in frame:
            continue
        pair = tuple(sorted(("ego", vid)))
        if pair in seen:
            continue
        if scene_gen.rectangles_overlap(frame["ego"], frame[vid]):
            pairs.append(pair)
    return pairs


def apply_waypoint_lane_intent(modified: np.ndarray, waypoints: np.ndarray) -> None:
    """Map a corrected local trajectory's lateral offset to a SUMO lane change."""
    if len(modified) == 0 or len(waypoints) == 0:
        return
    lateral = float(np.median(modified[:min(5, len(modified)), 1] - waypoints[:min(5, len(waypoints)), 1]))
    if abs(lateral) < 0.8:
        return
    try:
        road_id = traci.vehicle.getRoadID("ego")
        if not road_id or road_id.startswith(":"):
            return
        n_lanes = traci.edge.getLaneNumber(road_id)
        if n_lanes <= 1:
            return
        lane = traci.vehicle.getLaneIndex("ego")
        target = lane + (1 if lateral > 0 else -1)
        target = max(0, min(n_lanes - 1, target))
        if target != lane:
            traci.vehicle.changeLane("ego", target, 1.0)
    except Exception:
        pass


def apply_rear_escape_lane() -> bool:
    """Move ego to an adjacent lane when rear-impact risk dominates."""
    try:
        road_id = traci.vehicle.getRoadID("ego")
        if not road_id or road_id.startswith(":"):
            return False
        lane = traci.vehicle.getLaneIndex("ego")
        n_lanes = traci.edge.getLaneNumber(road_id)
        candidates = [lane + 1, lane - 1]
        candidates = [c for c in candidates if 0 <= c < n_lanes]
        if not candidates:
            return False

        ego_pos = traci.vehicle.getLanePosition("ego")
        best_lane = None
        best_clearance = -1.0
        for cand in candidates:
            clearance = 999.0
            lane_id = f"{road_id}_{cand}"
            for vid in traci.vehicle.getIDList():
                if vid == "ego":
                    continue
                try:
                    if traci.vehicle.getLaneID(vid) != lane_id:
                        continue
                    gap = abs(traci.vehicle.getLanePosition(vid) - ego_pos)
                    clearance = min(clearance, gap)
                except Exception:
                    continue
            if clearance > best_clearance:
                best_lane = cand
                best_clearance = clearance
        if best_lane is not None and best_clearance > 25.0:
            traci.vehicle.changeLane("ego", best_lane, 2.0)
            return True
    except Exception:
        pass
    return False


def run_scenario(
    net_file: Path,
    rou_file: Path,
    scenario_name: str,
    method,
    ego_uses_coop: bool,
    max_steps: int = 220,
) -> RunMetrics:
    cmd = [
        scene_gen.SUMO_BIN, "-n", str(net_file), "-r", str(rou_file),
        "--collision.action", "warn",
        "--collision.check-junctions", "true",
        "--collision.mingap-factor", "0",
        "--step-length", "0.1",
        "--no-step-log", "true",
        "--no-warnings", "true",
    ]
    metrics = RunMetrics()
    seen_pairs = set()
    seen_overlap_pairs = set()
    try:
        traci.start(cmd)
        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            apply_scene_controls(scenario_name, step, vids)
            if "ego" not in vids:
                if step > 5:
                    break
                continue
            try:
                traci.vehicle.setSpeedMode("ego", 0)
            except Exception:
                pass

            metrics.n_frames += 1
            perception = build_perception(ego_uses_coop=ego_uses_coop)
            ttc = estimate_min_ttc(perception)
            dangerous = ttc < 3.0
            ego_warned = False
            target_speed = None

            if perception and method is not None:
                waypoints = np.array([[max(perception.ego.velocity, 1.0) * (i + 1) * 0.5, 0.0]
                                      for i in range(10)])
                try:
                    if hasattr(method, "constrain_waypoints"):
                        modified, stats = method.constrain_waypoints(waypoints, perception)
                        apply_waypoint_lane_intent(modified, waypoints)
                        if stats.get("lane_escape", 0):
                            if not apply_rear_escape_lane():
                                stats["target_speed_factor"] = min(
                                    stats.get("target_speed_factor", 1.0), 0.8
                                )
                        fired = (
                            stats.get("n_geometric_threats", 0) > 0
                            or stats.get("n_collisions_detected", 0) > 0
                            or stats.get("n_modifications", 0) > 0
                        )
                        if fired:
                            ego_warned = True
                            target_speed = perception.ego.velocity * stats.get("target_speed_factor", 0.5)
                    else:
                        safe = method.constrain(perception)
                        if safe.mode == ConstraintMode.MINIMUM_HARM:
                            ego_warned = True
                            target_speed = perception.ego.velocity * 0.2
                        elif safe.mode == ConstraintMode.CONSERVATIVE:
                            ego_warned = True
                            target_speed = perception.ego.velocity * 0.6
                except Exception:
                    pass

            if target_speed is not None:
                try:
                    traci.vehicle.setSpeed("ego", max(0.0, target_speed))
                except Exception:
                    pass

            if perception:
                base_wp = np.array([[max(perception.ego.velocity, 1.0) * (i + 1) * 0.5, 0.0]
                                    for i in range(10)])
                check_wp = base_wp
                if method is not None and hasattr(method, "constrain_waypoints"):
                    try:
                        check_wp, _ = method.constrain_waypoints(base_wp, perception)
                    except Exception:
                        check_wp = base_wp
                for i in range(10):
                    dt = (i + 1) * 0.5
                    for agent in perception.agents:
                        ax = agent.state.x + agent.state.vx * dt
                        ay = agent.state.y + agent.state.vy * dt
                        if math.hypot(check_wp[i, 0] - ax, check_wp[i, 1] - ay) < 2.0:
                            metrics.wp_coll += 1
                            break
                metrics.wp_total += 10

            if dangerous:
                metrics.n_dangerous_frames += 1
                metrics.n_warned_dangerous += int(ego_warned)
            else:
                metrics.n_safe_frames += 1
                metrics.n_warned_safe += int(ego_warned)
            if ego_warned:
                metrics.n_warning_frames += 1
                metrics.n_modified_frames += 1
                if metrics.first_warning < 0:
                    metrics.first_warning = step

            frame = current_frame(vids)

            try:
                for col in traci.simulation.getCollisions():
                    pair = tuple(sorted((col.collider, col.victim)))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    if "ego" not in pair:
                        continue
                    metrics.unique_ego_coll += 1
                    metrics.collision_pairs.append("+".join(pair))
                    if metrics.first_collision < 0:
                        metrics.first_collision = step
                    else:
                        metrics.secondary_coll += 1
                    a, b = pair
                    va = frame[a][4] if a in frame else 0.0
                    vb = frame[b][4] if b in frame else 0.0
                    metrics.severities.append(abs(va - vb))
            except Exception:
                pass

            for pair in visual_collision_pairs(frame, seen_overlap_pairs):
                seen_overlap_pairs.add(pair)
                metrics.overlap_coll += 1
                if metrics.first_overlap < 0:
                    metrics.first_overlap = step

            if traci.simulation.getMinExpectedNumber() == 0:
                break
        traci.close()
    except Exception:
        try:
            traci.close()
        except Exception:
            pass
    return metrics


def scenario_defs():
    return [
        ("BS1_blind", *scene_gen.make_BS1(True)),
        ("BS1_nonblind", *scene_gen.make_BS1(False)),
        ("BS2_blind", *scene_gen.make_BS2(True)),
        ("BS2_nonblind", *scene_gen.make_BS2(False)),
        ("BS3_blind", *scene_gen.make_BS3(True)),
        ("BS3_nonblind", *scene_gen.make_BS3(False)),
        ("BS4_blind", *scene_gen.make_BS4(True)),
        ("BS4_nonblind", *scene_gen.make_BS4(False)),
        ("BS5_blind", *scene_gen.make_BS5(True)),
        ("BS5_nonblind", *scene_gen.make_BS5(False)),
    ]


def inject_rear_attacker(rou_file: Path, scenario_name: str) -> Path:
    """Add an aggressive rear vehicle to test secondary-collision risk."""
    out = rou_file.with_name(f"{rou_file.stem}_stress.rou.xml")
    text = rou_file.read_text()
    if "attacker_rear" in text:
        out.write_text(text)
        return out

    if scenario_name.startswith("BS1"):
        route, lane, depart, speed = "A1B1 B1C1", 1, 0.8, 13.5
    elif scenario_name.startswith("BS2"):
        route, lane, depart, speed = "B0C0 C0D0", 1, 0.7, 13.5
    elif scenario_name.startswith("BS3"):
        route, lane, depart, speed = "B3A1 A1B1", 0, 0.8, 13.5
    elif scenario_name.startswith("BS4"):
        route, lane, depart, speed = "B3A1 A1B1", 1, 0.8, 13.5
    else:
        route, lane, depart, speed = "A0B0 B0C0 C0D0 D0E0", 1, 2.8, 13.5

    rear = (
        f'    <vehicle id="attacker_rear" type="attacker" depart="{depart}" '
        f'departSpeed="{speed}" departLane="{lane}">\n'
        f'        <route edges="{route}"/>\n'
        f'    </vehicle>\n'
    )
    text = text.replace("</routes>", rear + "</routes>")
    out.write_text(text)
    return out


def stress_scenario_defs():
    scenarios = []
    for name, net_file, rou_file in scenario_defs():
        if name.startswith("BS5"):
            continue
        scenarios.append((f"{name}_stress", net_file, inject_rear_attacker(rou_file, name)))
    return scenarios


def method_configs(v1):
    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5, base_margin_invisible=4.0)
    configs = [
        ("NoCon-egoonly", lambda: None, False),
        ("NoCon-coop", lambda: None, True),
        ("RSS-coop", lambda: RSSOnly(), True),
        ("UniE2EV2X-coop", lambda: UniE2EV2XSafety(safety_threshold=3.0), True),
        ("MAP-coop", lambda: MAPSafety(min_clearance=0.5), True),
        ("RiskMM-coop", lambda: RiskMMSafety(v_max=20.0), True),
    ]
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid-thr{thr:.2f}",
                        lambda thr=thr: HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
                        True))
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid+AND-thr{thr:.2f}",
                        lambda thr=thr: HybridWithGeometricAND(
                            HybridSafetyConstraint(detection_threshold=thr, **base_kwargs)),
                        True))
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid+AND+TTC-thr{thr:.2f}",
                        lambda thr=thr: HybridWithGeometricANDTTC(
                            HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
                            ttc_override=3.0),
                        True))
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid+AND+TTC+RearAware-thr{thr:.2f}",
                        lambda thr=thr: HybridWithGeometricANDTTCAware(
                            HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
                            ttc_override=3.0),
                        True))
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid+AND+TTC+MinHarm-thr{thr:.2f}",
                        lambda thr=thr: HybridWithGeometricANDTTCMinHarm(
                            HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
                            ttc_override=3.0,
                            rear_gap_guard=18.0),
                        True))
    for thr in (0.25, 0.30, 0.40):
        configs.append((f"Hybrid+AND+TTC+RearEscape-thr{thr:.2f}",
                        lambda thr=thr: HybridWithGeometricANDTTCRearEscape(
                            HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
                            ttc_override=3.0,
                            rear_gap_guard=18.0),
                        True))
    return configs


def add_to_agg(agg: Agg, run: RunMetrics, truth_collision: bool, truth_collision_step: int) -> None:
    agg.n_runs += 1
    agg.ego_coll_total += run.unique_ego_coll
    agg.secondary_total += run.secondary_coll
    agg.severities.extend(run.severities)
    agg.n_frames += run.n_frames
    agg.n_dangerous_frames += run.n_dangerous_frames
    agg.n_warned_dangerous += run.n_warned_dangerous
    agg.n_safe_frames += run.n_safe_frames
    agg.n_warned_safe += run.n_warned_safe
    agg.wp_coll += run.wp_coll
    agg.wp_total += run.wp_total
    agg.n_modified_frames += run.n_modified_frames
    agg.overlap_total += run.overlap_coll
    if run.overlap_coll > 0:
        agg.overlap_runs += 1
    if run.unique_ego_coll > 0:
        agg.coll_runs += 1
    if truth_collision:
        agg.n_collision_truth += 1
        if run.first_warning >= 0 and run.first_warning <= truth_collision_step:
            agg.n_detect_scen += 1
            agg.early_warnings.append(truth_collision_step - run.first_warning)
    else:
        agg.n_normal_truth += 1
        if run.first_warning >= 0:
            agg.n_fa_scen += 1


def summarize(agg: Agg) -> dict[str, float]:
    return {
        "runs": agg.n_runs,
        "CollRate": agg.coll_runs / max(agg.n_runs, 1),
        "2ndC": agg.secondary_total,
        "Sev": float(np.mean(agg.severities)) if agg.severities else 0.0,
        "Det(s)": agg.n_detect_scen / max(agg.n_collision_truth, 1),
        "Det(f)": agg.n_warned_dangerous / max(agg.n_dangerous_frames, 1),
        "Early": float(np.mean(agg.early_warnings)) if agg.early_warnings else 0.0,
        "FA(s)": agg.n_fa_scen / max(agg.n_normal_truth, 1),
        "FA(f)": agg.n_warned_safe / max(agg.n_safe_frames, 1),
        "WPC%": agg.wp_coll / max(agg.wp_total, 1),
        "Mod%": agg.n_modified_frames / max(agg.n_frames, 1),
        "OverlapRate": agg.overlap_runs / max(agg.n_runs, 1),
        "OverlapN": agg.overlap_total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-set", choices=["base", "stress", "all"], default="base")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--key-methods", action="store_true",
                        help="Run a smaller method set for quick stress validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ROOT / "models" / "collision_net_best.pt", map_location="cpu", weights_only=False)
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt["model"])
    v1.eval()

    if args.scenario_set == "base":
        scenarios = scenario_defs()
    elif args.scenario_set == "stress":
        scenarios = stress_scenario_defs()
    else:
        scenarios = scenario_defs() + stress_scenario_defs()

    methods = method_configs(v1)
    if args.key_methods:
        keep = {
            "NoCon-egoonly", "NoCon-coop", "RSS-coop", "UniE2EV2X-coop", "MAP-coop",
            "RiskMM-coop", "Hybrid-thr0.30", "Hybrid+AND-thr0.30",
            "Hybrid+AND+TTC-thr0.30", "Hybrid+AND+TTC+RearAware-thr0.30",
            "Hybrid+AND+TTC+MinHarm-thr0.30", "Hybrid+AND+TTC+RearEscape-thr0.30",
        }
        methods = [m for m in methods if m[0] in keep]

    print("=" * 72)
    print("  Revised SUMO 10-Scenario Comparison")
    print("=" * 72)
    print(f"  scenario_set={args.scenario_set}, scenarios={len(scenarios)}, methods={len(methods)}")

    truth = {}
    for sname, net_file, rou_file in scenarios:
        run = run_scenario(net_file, rou_file, sname, None, ego_uses_coop=True)
        truth[sname] = run.first_collision
        print(f"  truth {sname:14s}: collision_step={run.first_collision}")

    results = defaultdict(Agg)
    per_run_rows = []
    t0 = time.time()
    for sname, net_file, rou_file in scenarios:
        truth_step = truth[sname]
        truth_collision = truth_step >= 0
        print(f"\n  --- {sname} ---")
        for mname, factory, uses_coop in methods:
            run = run_scenario(net_file, rou_file, sname, factory(), uses_coop)
            add_to_agg(results[(mname, sname)], run, truth_collision, truth_step)
            add_to_agg(results[(mname, "OVERALL")], run, truth_collision, truth_step)
            per_run_rows.append({
                "scenario": sname,
                "method": mname,
                "truth_collision_step": truth_step,
                "collision": int(run.unique_ego_coll > 0),
                "first_collision": run.first_collision,
                "overlap": int(run.overlap_coll > 0),
                "collision_pairs": ";".join(run.collision_pairs),
                "first_overlap": run.first_overlap,
                "overlap_count": run.overlap_coll,
                "first_warning": run.first_warning,
                "severity": np.mean(run.severities) if run.severities else 0.0,
                "wp_coll": run.wp_coll,
                "wp_total": run.wp_total,
                "warning_frames": run.n_warning_frames,
                "frames": run.n_frames,
            })
            print(f"    {mname:23s}: coll={run.unique_ego_coll > 0} "
                  f"warn={run.first_warning:4d} wpc={run.wp_coll / max(run.wp_total, 1):.1%} "
                  f"overlap={run.overlap_coll}")

    rows = []
    for scenario in [s[0] for s in scenarios] + ["OVERALL"]:
        print(f"\n{'=' * 72}")
        print(f"  {scenario}")
        print("=" * 72)
        print(f"  {'Method':23s} {'CollRate':>8s} {'2ndC':>4s} {'Sev':>5s} "
              f"{'Det(s)':>7s} {'Det(f)':>7s} {'Early':>6s} "
              f"{'FA(s)':>6s} {'FA(f)':>6s} {'WPC%':>6s} {'Mod%':>6s} {'OvR':>6s}")
        for mname, _, _ in methods:
            stats = summarize(results[(mname, scenario)])
            rows.append({"scenario": scenario, "method": mname, **stats})
            print(f"  {mname:23s} {stats['CollRate']:8.0%} {int(stats['2ndC']):4d} "
                  f"{stats['Sev']:5.2f} {stats['Det(s)']:6.0%} {stats['Det(f)']:6.0%} "
                  f"{stats['Early']:5.1f} {stats['FA(s)']:5.0%} {stats['FA(f)']:5.0%} "
                  f"{stats['WPC%']:5.1%} {stats['Mod%']:5.0%} {stats['OverlapRate']:5.0%}")

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    per_run_csv = out_dir / "per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)

    print(f"\n  Wrote: {summary_csv}")
    print(f"  Wrote: {per_run_csv}")
    print(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

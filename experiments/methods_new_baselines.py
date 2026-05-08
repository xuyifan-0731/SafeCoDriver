"""Three new baseline safety constraint methods for comparison.

1. UniE2EV2X [Li et al., 2024] — Geometric collision detection post-processing
2. MAP [Yin et al., 2025] — Collision-aware bbox checking + map feasibility
3. RiskMM [Lei et al., 2025] — Learning-based risk map + MPC constraint

All methods share the same interface:
  Input: predicted waypoints (B, T, 2) + perception (agents + ego)
  Output: safety-modified waypoints (B, T, 2)

This allows fair comparison when plugged into the same CoDriving Planner.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafeActionSpace, ConstraintMode,
)
from coop_safety.perception.dynamics import BicycleModel, get_dynamics_params


# ============================================================
# 1. UniE2EV2X: Geometric Collision Post-Processing
# ============================================================

class UniE2EV2XSafety:
    """UniE2EV2X [Li et al., 2024] safety constraint.

    Ref: "Unified End-to-End V2X Cooperative Autonomous Driving"
    arXiv: 2405.03971

    Mechanism: After predicting motion trajectories, check frame-by-frame
    if the minimum distance between ego polygon and any agent polygon
    is below a safety threshold. If collision detected, push ego waypoint
    away from the colliding agent.

    This is a post-processing step — does not change the planning model.
    """

    name = "UniE2EV2X [Li24]"

    def __init__(self, safety_threshold: float = 3.0, push_distance: float = 2.0):
        """
        Args:
            safety_threshold: minimum distance (m) below which collision is detected
            push_distance: how far to push waypoint away from colliding agent (m)
        """
        self.safety_threshold = safety_threshold
        self.push_distance = push_distance

    def constrain_waypoints(self, waypoints: np.ndarray,
                            perception: PerceptionResult) -> tuple[np.ndarray, dict]:
        """Check and modify waypoints using geometric collision detection.

        Args:
            waypoints: (T, 2) predicted ego waypoints in ego frame
            perception: current scene perception

        Returns:
            (modified_waypoints, stats_dict)
        """
        modified = waypoints.copy()
        n_collisions = 0
        ego = perception.ego

        for t in range(len(waypoints)):
            wp = waypoints[t]

            # Predict agent position at time t (constant velocity)
            dt = (t + 1) * 0.5  # Assume 0.5s per waypoint step
            for agent in perception.agents:
                a = agent.state
                # Predicted agent position
                ax = a.x + a.vx * dt
                ay = a.y + a.vy * dt

                # Build ego polygon at waypoint
                ego_poly = self._make_box(wp[0], wp[1], ego.heading, ego.length, ego.width)
                # Build agent polygon at predicted position
                agent_poly = self._make_box(ax, ay, a.heading, a.length, a.width)

                # Check minimum distance
                try:
                    dist = ego_poly.distance(agent_poly)
                except:
                    dist = math.sqrt((wp[0]-ax)**2 + (wp[1]-ay)**2)

                if dist < self.safety_threshold:
                    n_collisions += 1
                    # Push waypoint away from agent
                    dx = wp[0] - ax
                    dy = wp[1] - ay
                    d = max(math.sqrt(dx**2 + dy**2), 0.1)
                    push_x = dx / d * self.push_distance
                    push_y = dy / d * self.push_distance
                    modified[t, 0] += push_x
                    modified[t, 1] += push_y
                    break  # Only handle closest collision per timestep

        stats = {
            "method": self.name,
            "n_collisions_detected": n_collisions,
            "modification_rate": n_collisions / max(len(waypoints), 1),
            "n_geometric_threats": n_collisions,
        }
        return modified, stats

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Standard SafeActionSpace interface (for standalone evaluation)."""
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])
        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)

        # Remove collision zones
        feasible = Polygon(boundary) if len(boundary) >= 3 else Point(ego.x, ego.y).buffer(30)
        if not feasible.is_valid:
            feasible = feasible.buffer(0)

        for agent in perception.agents:
            a = agent.state
            for dt in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
                ax = a.x + a.vx * dt
                ay = a.y + a.vy * dt
                agent_zone = Point(ax, ay).buffer(self.safety_threshold)
                candidate = feasible.difference(agent_zone)
                if hasattr(candidate, 'area') and candidate.area > 5:
                    feasible = candidate

        from shapely.geometry import MultiPolygon
        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') and not feasible.is_empty else boundary
        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=ConstraintMode.NORMAL,
            safety_margin_ttc=float('inf'),
            reasoning=[f"UniE2EV2X: geometric collision check, threshold={self.safety_threshold}m"],
        )

    @staticmethod
    def _make_box(x, y, heading, length, width):
        """Create oriented bounding box polygon."""
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        hl, hw = length / 2, width / 2
        corners = [
            (x + cos_h*hl - sin_h*hw, y + sin_h*hl + cos_h*hw),
            (x + cos_h*hl + sin_h*hw, y + sin_h*hl - cos_h*hw),
            (x - cos_h*hl + sin_h*hw, y - sin_h*hl - cos_h*hw),
            (x - cos_h*hl - sin_h*hw, y - sin_h*hl + cos_h*hw),
        ]
        return Polygon(corners)


# ============================================================
# 2. MAP: Collision-Aware BBox Checking + Map Feasibility
# ============================================================

class MAPSafety:
    """MAP [Yin et al., 2025] safety constraint.

    Ref: "MAP: End-to-End Autonomous Driving with Map-Assisted Planning"
    arXiv: 2509.13926

    Mechanism: For each predicted waypoint, construct ego bounding box and
    check intersection with GT obstacle bounding boxes. If collision detected,
    apply minimum displacement to resolve. Also constrains to road boundary
    (if available).

    Mirrors the Collision Loss used during training, applied as post-processing.
    """

    name = "MAP [Yin25]"

    def __init__(self, ego_length: float = 4.5, ego_width: float = 1.8,
                 min_clearance: float = 0.5):
        self.ego_length = ego_length
        self.ego_width = ego_width
        self.min_clearance = min_clearance

    def constrain_waypoints(self, waypoints: np.ndarray,
                            perception: PerceptionResult) -> tuple[np.ndarray, dict]:
        """Check and modify waypoints using bbox collision detection."""
        modified = waypoints.copy()
        ego = perception.ego
        n_collisions = 0

        for t in range(len(waypoints)):
            wp = modified[t]
            dt = (t + 1) * 0.5

            ego_box = UniE2EV2XSafety._make_box(
                wp[0], wp[1], ego.heading, self.ego_length, self.ego_width)
            # Buffer for clearance
            ego_buffered = ego_box.buffer(self.min_clearance)

            for agent in perception.agents:
                a = agent.state
                ax = a.x + a.vx * dt
                ay = a.y + a.vy * dt
                agent_box = UniE2EV2XSafety._make_box(
                    ax, ay, a.heading, a.length, a.width)

                if ego_buffered.intersects(agent_box):
                    n_collisions += 1
                    # Minimum displacement to resolve collision
                    overlap = ego_buffered.intersection(agent_box)
                    if not overlap.is_empty:
                        # Push in direction away from agent center
                        agent_center = np.array([ax, ay])
                        ego_center = np.array([wp[0], wp[1]])
                        direction = ego_center - agent_center
                        d = np.linalg.norm(direction)
                        if d > 0.01:
                            direction /= d
                            # Push by overlap extent + clearance
                            push = math.sqrt(overlap.area) + self.min_clearance
                            modified[t, 0] += direction[0] * push
                            modified[t, 1] += direction[1] * push
                    break

        stats = {
            "method": self.name,
            "n_collisions_detected": n_collisions,
            "modification_rate": n_collisions / max(len(waypoints), 1),
            "n_geometric_threats": n_collisions,
        }
        return modified, stats

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Standard SafeActionSpace interface."""
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])
        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)
        feasible = Polygon(boundary) if len(boundary) >= 3 else Point(ego.x, ego.y).buffer(30)
        if not feasible.is_valid:
            feasible = feasible.buffer(0)

        # Remove agent bbox + clearance zones
        for agent in perception.agents:
            a = agent.state
            for dt in [0.5, 1.0, 1.5, 2.0]:
                ax = a.x + a.vx * dt
                ay = a.y + a.vy * dt
                agent_zone = UniE2EV2XSafety._make_box(
                    ax, ay, a.heading, a.length + self.min_clearance*2,
                    a.width + self.min_clearance*2)
                try:
                    candidate = feasible.difference(agent_zone)
                    from shapely.geometry import MultiPolygon
                    if isinstance(candidate, MultiPolygon):
                        candidate = max(candidate.geoms, key=lambda g: g.area)
                    if candidate.area > 5:
                        feasible = candidate
                except:
                    pass

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') and not feasible.is_empty else boundary
        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=ConstraintMode.NORMAL,
            safety_margin_ttc=float('inf'),
            reasoning=[f"MAP: bbox collision check + {self.min_clearance}m clearance"],
        )


# ============================================================
# 3. RiskMM: Learning-based Risk Map + Simplified MPC
# ============================================================

class RiskMMSafety:
    """RiskMM [Lei et al., 2025] safety constraint.

    Ref: "Risk Map As Middleware: Towards Interpretable Cooperative
    End-to-end Autonomous Driving for Risk-Aware Planning"
    arXiv: 2508.07686

    Mechanism: Build a risk map from agent interaction attention,
    then use simplified MPC to optimize trajectory under risk + dynamics
    constraints.

    Simplified implementation:
    - Risk map: attention-weighted agent interaction (like our RiskMap but learned)
    - MPC: quadratic cost optimization with speed constraint
    """

    name = "RiskMM [Lei25]"

    def __init__(self, v_max: float = 20.0, risk_weight: float = 1.0,
                 smooth_weight: float = 0.5):
        self.v_max = v_max  # m/s (≈72 km/h)
        self.risk_weight = risk_weight
        self.smooth_weight = smooth_weight

    def constrain_waypoints(self, waypoints: np.ndarray,
                            perception: PerceptionResult) -> tuple[np.ndarray, dict]:
        """Optimize waypoints using risk-weighted MPC-style cost.

        For each waypoint, compute:
          cost = risk_weight × risk(x,y) + smooth_weight × |Δwp|²
        Then gradient-descend to minimize cost while respecting speed constraint.
        """
        modified = waypoints.copy()
        ego = perception.ego

        # Build simple risk map: sum of Gaussian fields around agents
        def risk_at(x, y, t):
            dt = (t + 1) * 0.5
            total_risk = 0.0
            for agent in perception.agents:
                a = agent.state
                ax = a.x + a.vx * dt
                ay = a.y + a.vy * dt
                dist_sq = (x - ax)**2 + (y - ay)**2
                # Gaussian risk field with speed-dependent spread
                sigma = max(a.length, 3.0) + max(a.velocity, 1.0) * 0.5
                total_risk += math.exp(-dist_sq / (2 * sigma**2))
            return total_risk

        # Simple gradient-based MPC optimization (5 iterations)
        lr = 0.3
        n_modifications = 0

        for iteration in range(5):
            for t in range(len(modified)):
                x, y = modified[t]
                risk = risk_at(x, y, t)

                if risk < 0.1:
                    continue  # Low risk, no need to modify

                n_modifications += 1

                # Numerical gradient of risk
                eps = 0.5
                grad_x = (risk_at(x + eps, y, t) - risk_at(x - eps, y, t)) / (2 * eps)
                grad_y = (risk_at(x, y + eps, t) - risk_at(x, y - eps, t)) / (2 * eps)

                # Smoothness: pull toward midpoint of neighbors
                if 0 < t < len(modified) - 1:
                    mid_x = (modified[t-1, 0] + modified[t+1, 0]) / 2
                    mid_y = (modified[t-1, 1] + modified[t+1, 1]) / 2
                    smooth_grad_x = self.smooth_weight * (x - mid_x)
                    smooth_grad_y = self.smooth_weight * (y - mid_y)
                else:
                    smooth_grad_x = smooth_grad_y = 0

                # Update
                modified[t, 0] -= lr * (self.risk_weight * grad_x + smooth_grad_x)
                modified[t, 1] -= lr * (self.risk_weight * grad_y + smooth_grad_y)

            # Speed constraint: clamp displacement between consecutive waypoints
            for t in range(1, len(modified)):
                dx = modified[t, 0] - modified[t-1, 0]
                dy = modified[t, 1] - modified[t-1, 1]
                dist = math.sqrt(dx**2 + dy**2)
                max_dist = self.v_max * 0.5  # max distance per 0.5s step
                if dist > max_dist:
                    scale = max_dist / dist
                    modified[t, 0] = modified[t-1, 0] + dx * scale
                    modified[t, 1] = modified[t-1, 1] + dy * scale

        stats = {
            "method": self.name,
            "n_modifications": n_modifications,
            "modification_rate": n_modifications / max(len(waypoints) * 5, 1),
            # Bug fix 260508: add standard fields for evaluation compatibility
            "n_collisions_detected": 1 if n_modifications > 0 else 0,
            "n_geometric_threats": n_modifications,
        }
        return modified, stats

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Standard SafeActionSpace interface."""
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])
        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)
        feasible = Polygon(boundary) if len(boundary) >= 3 else Point(ego.x, ego.y).buffer(30)
        if not feasible.is_valid:
            feasible = feasible.buffer(0)

        # Risk-weighted exclusion
        for agent in perception.agents:
            a = agent.state
            speed = max(a.velocity, 1.0)
            sigma = max(a.length, 3.0) + speed * 0.5
            risk_zone = Point(a.x, a.y).buffer(sigma * 1.5)
            try:
                candidate = feasible.difference(risk_zone)
                from shapely.geometry import MultiPolygon
                if isinstance(candidate, MultiPolygon):
                    candidate = max(candidate.geoms, key=lambda g: g.area)
                if candidate.area > 5:
                    feasible = candidate
            except:
                pass

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') and not feasible.is_empty else boundary
        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=min(params.max_speed, self.v_max),
            mode=ConstraintMode.NORMAL,
            safety_margin_ttc=float('inf'),
            reasoning=[f"RiskMM: risk-weighted MPC, v_max={self.v_max}m/s"],
        )

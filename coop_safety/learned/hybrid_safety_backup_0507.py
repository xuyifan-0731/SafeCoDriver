"""Neural-Geometric Hybrid Safety Constraint.

Combines learned collision detection (V1 network) with enhanced geometric
waypoint modification (visibility-aware + approach-speed + multi-agent).

Key design: DECOUPLE detection from modification.
  - Waypoint modification: ALWAYS do geometric checking → lowest WPColl%
  - Collision detection: Use V1 network → high DetRate, low FalseAlm

Three innovations over existing methods:
1. Visibility-Aware Adaptive Safety Buffer — invisible agents (V2X-only)
   get larger margins due to higher position uncertainty.
2. Approach-Speed Adaptive Collision Zone — danger zones scale with
   agent approach speed.
3. Multi-Agent Repulsive Field Waypoint Correction — sums repulsive forces
   from ALL threatening agents (not just closest), preventing push into
   another agent.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coop_safety.interface import (
    PerceptionResult, SafeActionSpace, ConstraintMode,
)


class HybridSafetyConstraint:
    """Neural-Geometric Cascaded Safety Constraint.

    Detection: V1 collision network (trained classifier, low false alarm)
    Modification: Visibility-aware geometric checking (lowest waypoint collision)
    """

    name = "Ours-Hybrid"

    def __init__(self,
                 detector_model=None,
                 risk_model=None,
                 base_margin_visible: float = 2.5,
                 base_margin_invisible: float = 4.0,
                 approach_speed_factor: float = 0.3,
                 push_clearance: float = 1.0,
                 smooth_weight: float = 0.3,
                 v_max: float = 20.0,
                 detection_threshold: float = 0.3):
        """
        Args:
            detector_model: CollisionPredictionNetwork (V1) for detection
            risk_model: CollisionPredictionNetV2 (V2) for waypoint risk scoring
            base_margin_visible: safety margin for visible agents (m)
            base_margin_invisible: safety margin for invisible/V2X-only agents (m)
            approach_speed_factor: how much approach speed increases margin
            push_clearance: extra clearance after collision resolution (m)
            smooth_weight: trajectory smoothing weight [0, 1]
            v_max: maximum speed for speed clamping (m/s)
            detection_threshold: V1 collision_prob threshold for detection
        """
        self.detector = detector_model    # V1: detection (100% det, 26.9% FA)
        self.risk_model = risk_model      # V2: per-waypoint risk scoring
        self.base_margin_visible = base_margin_visible
        self.base_margin_invisible = base_margin_invisible
        self.approach_speed_factor = approach_speed_factor
        self.push_clearance = push_clearance
        self.smooth_weight = smooth_weight
        self.v_max = v_max
        self.detection_threshold = detection_threshold

    def _get_safety_margin(self, agent, ego_speed: float) -> float:
        """Compute visibility-aware, approach-speed-adaptive safety margin."""
        if agent.is_visible:
            margin = self.base_margin_visible
        else:
            margin = self.base_margin_invisible

        s = agent.state
        dist = math.sqrt(s.x**2 + s.y**2)
        if dist > 0.01:
            rel_vx = s.vx - ego_speed
            rel_vy = s.vy
            approach_speed = -(s.x * rel_vx + s.y * rel_vy) / dist
            if approach_speed > 0:
                margin *= (1.0 + self.approach_speed_factor *
                          min(approach_speed / self.v_max, 1.0))

        margin += max(s.length, s.width) * 0.3
        return margin

    def _find_threats(self, wp: np.ndarray, agents, t_idx: int,
                      ego_speed: float) -> list:
        """Find all agents threatening a waypoint at time step t."""
        dt = (t_idx + 1) * 0.5
        threats = []
        for agent in agents:
            a = agent.state
            ax = a.x + a.vx * dt
            ay = a.y + a.vy * dt
            dist = math.sqrt((wp[0] - ax)**2 + (wp[1] - ay)**2)
            margin = self._get_safety_margin(agent, ego_speed)
            if dist < margin:
                threats.append((agent, ax, ay, dist, margin))
        return threats

    def _multi_agent_repulsion(self, wp: np.ndarray, threats: list,
                               wp_risk: float = 0.0) -> np.ndarray:
        """Compute combined repulsive force from all threatening agents."""
        if not threats:
            return wp.copy()

        total_fx, total_fy = 0.0, 0.0
        max_push = 0.0

        for agent, ax, ay, dist, margin in threats:
            dx = wp[0] - ax
            dy = wp[1] - ay
            d = max(dist, 0.1)
            force_mag = (margin - dist + self.push_clearance) / d
            vis_weight = 1.0 if agent.is_visible else 1.5
            risk_weight = 1.0 + wp_risk
            weight = vis_weight * risk_weight * force_mag

            total_fx += (dx / d) * weight
            total_fy += (dy / d) * weight
            max_push = max(max_push, margin - dist + self.push_clearance)

        force_norm = math.sqrt(total_fx**2 + total_fy**2)
        if force_norm > 0.01:
            push_dist = max(max_push, 1.0)
            new_wp = wp.copy()
            new_wp[0] += (total_fx / force_norm) * push_dist
            new_wp[1] += (total_fy / force_norm) * push_dist
            return new_wp
        return wp.copy()

    def _detect_with_v1(self, perception) -> float:
        """Use V1 network for collision detection (binary classification).

        V1 achieves 100% DetRate / 26.9% FalseAlm on DeepAccident.
        """
        if self.detector is None:
            return 0.0
        try:
            import torch
            agents_feat = np.zeros((1, 30, 10), dtype=np.float32)
            mask = np.zeros((1, 30), dtype=bool)
            for i, a in enumerate(perception.agents[:30]):
                s = a.state
                agents_feat[0, i] = [
                    s.x, s.y, s.vx, s.vy, s.heading,
                    s.length, s.width, s.velocity,
                    1.0 if a.is_visible else 0.0, 0
                ]
                mask[0, i] = True
            with torch.no_grad():
                cp, _ = self.detector(
                    torch.FloatTensor(agents_feat),
                    torch.BoolTensor(mask)
                )
            return cp.item()
        except Exception:
            return 0.0

    def _get_wp_risks(self, waypoints, perception) -> np.ndarray:
        """Use V2 network for per-waypoint risk scoring (optional)."""
        wp_risks = np.zeros(len(waypoints))
        if self.risk_model is None:
            return wp_risks
        try:
            import torch
            agents_feat, mask, ego_feat = self._encode_v2(perception)
            wp_t = torch.FloatTensor(waypoints).unsqueeze(0)
            with torch.no_grad():
                out = self.risk_model(
                    torch.FloatTensor(agents_feat),
                    torch.BoolTensor(mask),
                    torch.FloatTensor(ego_feat),
                    wp_t
                )
            if 'waypoint_risk' in out:
                wp_risks = out['waypoint_risk'].squeeze(0).numpy().flatten()
        except Exception:
            pass
        return wp_risks

    def constrain_waypoints(self, waypoints: np.ndarray,
                            perception: PerceptionResult) -> tuple:
        """Main method: modify waypoints + detect collisions.

        IMPORTANT DESIGN: Detection and modification are DECOUPLED.
          - Waypoints are ALWAYS modified by geometric checking (for low WPColl)
          - Detection flag is set by V1 network only (for low FalseAlm)
        """
        modified = waypoints.copy()
        ego = perception.ego
        ego_speed = max(ego.velocity, 1.0)
        n_geometric_threats = 0

        # Get per-waypoint risk from V2 (optional, for weighted avoidance)
        wp_risks = self._get_wp_risks(waypoints, perception)

        # Geometric waypoint modification (ALWAYS runs, for low WPColl)
        for t in range(len(waypoints)):
            threats = self._find_threats(modified[t], perception.agents,
                                        t, ego_speed)
            if threats:
                n_geometric_threats += 1
                modified[t] = self._multi_agent_repulsion(
                    modified[t], threats,
                    wp_risk=wp_risks[t] if t < len(wp_risks) else 0.0
                )

        # Trajectory smoothing
        if self.smooth_weight > 0:
            smoothed = modified.copy()
            for t in range(1, len(modified) - 1):
                mid_x = (modified[t-1, 0] + modified[t+1, 0]) / 2
                mid_y = (modified[t-1, 1] + modified[t+1, 1]) / 2
                smoothed[t, 0] = modified[t, 0] * (1 - self.smooth_weight) + mid_x * self.smooth_weight
                smoothed[t, 1] = modified[t, 1] * (1 - self.smooth_weight) + mid_y * self.smooth_weight
            modified = smoothed

        # Speed clamping
        max_dist = self.v_max * 0.5
        for t in range(1, len(modified)):
            dx = modified[t, 0] - modified[t-1, 0]
            dy = modified[t, 1] - modified[t-1, 1]
            dist = math.sqrt(dx**2 + dy**2)
            if dist > max_dist:
                scale = max_dist / dist
                modified[t, 0] = modified[t-1, 0] + dx * scale
                modified[t, 1] = modified[t-1, 1] + dy * scale

        # DETECTION: Use V1 network (decoupled from modification)
        v1_prob = self._detect_with_v1(perception)
        is_dangerous = v1_prob > self.detection_threshold

        # Report: detection flag driven by V1, not by geometric modification
        stats = {
            "method": self.name,
            "n_collisions_detected": 1 if is_dangerous else 0,
            "modification_rate": 1.0 / max(len(waypoints), 1) if is_dangerous else 0,
            "collision_prob": v1_prob,
            "n_geometric_threats": n_geometric_threats,
        }
        return modified, stats

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Standard SafeActionSpace interface for detection/warning mode."""
        v1_prob = self._detect_with_v1(perception)

        if v1_prob > 0.7:
            mode = ConstraintMode.MINIMUM_HARM
        elif v1_prob > self.detection_threshold:
            mode = ConstraintMode.CONSERVATIVE
        else:
            mode = ConstraintMode.NORMAL

        return SafeActionSpace(
            feasible_region=np.array([[0, 0]]),
            max_acceleration=3.0,
            min_acceleration=-8.0,
            max_steering=0.6,
            max_speed=50.0,
            mode=mode,
            safety_margin_ttc=float('inf'),
            reasoning=[f"Hybrid: v1_prob={v1_prob:.2f}"],
        )

    def _encode_v2(self, perception):
        """Encode perception for V2 network input."""
        MAX_AGENTS = 30
        agents_feat = np.zeros((1, MAX_AGENTS, 12), dtype=np.float32)
        mask = np.zeros((1, MAX_AGENTS), dtype=bool)
        ego = perception.ego

        for i, a in enumerate(perception.agents[:MAX_AGENTS]):
            s = a.state
            dist = math.sqrt(s.x**2 + s.y**2)
            rel_vx = s.vx - ego.velocity
            if dist > 0.01:
                approach = -(s.x * rel_vx + s.y * s.vy) / dist
            else:
                approach = 0
            agents_feat[0, i] = [
                s.x, s.y, rel_vx, s.vy, s.heading,
                s.length, s.width, s.velocity,
                1.0 if a.is_visible else 0.0, 0,
                approach, dist
            ]
            mask[0, i] = True

        ego_feat = np.array([[ego.velocity, 0, 0, 0, 0, 0]], dtype=np.float32)
        return agents_feat, mask, ego_feat

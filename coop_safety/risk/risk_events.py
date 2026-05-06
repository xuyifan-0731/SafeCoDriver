from __future__ import annotations
"""RiskEvents: Concrete collision event enumeration.

Instantiates specific potential collision scenarios from RiskGraph edges,
including collision type classification, severity estimation, and probability.
Corresponds to research plan step 2.6.
"""

import numpy as np
from typing import Optional

from ..interface import (
    Agent, VehicleState, ConflictEdge, CollisionEvent,
)


class RiskEventEnumerator:
    """Enumerate and characterize potential collision events.

    RiskEvents differs from RiskGraph in that it produces concrete, actionable
    collision scenarios rather than abstract pairwise relationships:
    - RiskGraph: "Agent A and B have TTC=2.1s, collision_prob=0.6"
    - RiskEvents: "Rear-end collision between A and B in lane 2, t=[1.8,2.5]s,
                   severity=0.7 (high speed delta), probability=0.6"
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.min_probability = cfg.get("min_probability", 0.1)
        self.severity_speed_weight = cfg.get("severity_speed_weight", 0.4)
        self.severity_mass_weight = cfg.get("severity_mass_weight", 0.3)
        self.severity_angle_weight = cfg.get("severity_angle_weight", 0.3)

    def enumerate(self, conflict_edges: list[ConflictEdge],
                  agents_by_id: dict[str, Agent],
                  ego: VehicleState) -> list[CollisionEvent]:
        """Enumerate collision events from conflict edges.

        Args:
            conflict_edges: Pairwise conflicts from RiskGraph
            agents_by_id: Dict of agent_id → Agent
            ego: Ego vehicle state

        Returns:
            List of CollisionEvent, sorted by severity * probability
        """
        events = []

        for i, edge in enumerate(conflict_edges):
            if edge.collision_probability < self.min_probability:
                continue

            state_a = self._get_state(edge.agent_a_id, agents_by_id, ego)
            state_b = self._get_state(edge.agent_b_id, agents_by_id, ego)
            if state_a is None or state_b is None:
                continue

            # Classify collision type
            collision_type = self._classify_collision(state_a, state_b)

            # Estimate severity
            severity = self._estimate_severity(state_a, state_b, collision_type)

            # Build spatial region around collision point
            if edge.collision_point is not None:
                region = self._collision_region(edge.collision_point, state_a, state_b)
            else:
                mid = np.array([(state_a.x + state_b.x) / 2,
                                (state_a.y + state_b.y) / 2])
                region = self._collision_region(mid, state_a, state_b)

            events.append(CollisionEvent(
                event_id=f"evt_{i:03d}",
                participants=[edge.agent_a_id, edge.agent_b_id],
                spatial_region=region,
                time_window=edge.time_window,
                collision_type=collision_type,
                severity=severity,
                probability=edge.collision_probability,
            ))

        # Sort by risk = severity * probability (descending)
        events.sort(key=lambda e: e.severity * e.probability, reverse=True)
        return events

    def _classify_collision(self, a: VehicleState, b: VehicleState) -> str:
        """Classify collision type based on relative heading and position."""
        heading_diff = abs(self._normalize_angle(a.heading - b.heading))

        # Relative position angle
        dx = b.x - a.x
        dy = b.y - a.y
        rel_angle = abs(self._normalize_angle(np.arctan2(dy, dx) - a.heading))

        if b.vehicle_type == "pedestrian":
            return "pedestrian"

        if heading_diff < np.radians(30):
            # Similar heading
            if rel_angle < np.radians(30):
                return "rear_end"
            elif rel_angle > np.radians(150):
                return "rear_end"  # B is behind A
            else:
                return "side"
        elif heading_diff > np.radians(150):
            return "head_on"
        elif np.radians(60) < heading_diff < np.radians(120):
            return "intersection"  # Roughly perpendicular
        else:
            return "side"

    def _estimate_severity(self, a: VehicleState, b: VehicleState,
                           collision_type: str) -> float:
        """Estimate collision severity [0, 1].

        Based on:
        - Speed differential (higher = more severe)
        - Mass ratio (heavier hitting lighter = more severe for lighter)
        - Collision angle (head-on > side > rear-end)
        """
        # Speed component: relative closing speed
        rel_vx = a.vx - b.vx
        rel_vy = a.vy - b.vy
        closing_speed = np.sqrt(rel_vx ** 2 + rel_vy ** 2)
        speed_severity = min(1.0, closing_speed / 30.0)  # Normalize by 30 m/s (~108 km/h)

        # Mass component: mass asymmetry
        mass_ratio = max(a.mass, b.mass) / max(min(a.mass, b.mass), 100)
        mass_severity = min(1.0, (mass_ratio - 1) / 5.0)  # Big ratio = worse for lighter

        # Pedestrian always max mass severity
        if a.vehicle_type == "pedestrian" or b.vehicle_type == "pedestrian":
            mass_severity = 1.0

        # Angle component by collision type
        angle_severity = {
            "head_on": 1.0,
            "intersection": 0.8,
            "side": 0.6,
            "rear_end": 0.4,
            "pedestrian": 0.9,
        }.get(collision_type, 0.5)

        severity = (self.severity_speed_weight * speed_severity +
                    self.severity_mass_weight * mass_severity +
                    self.severity_angle_weight * angle_severity)
        return min(1.0, severity)

    def _collision_region(self, center: np.ndarray,
                          a: VehicleState, b: VehicleState) -> np.ndarray:
        """Build polygon region around predicted collision point.

        Uses vehicle dimensions to estimate the area affected.
        """
        # Region radius: max of both vehicle sizes + buffer
        radius = max(a.length, b.length) + 2.0

        # Simple circular approximation (8-sided polygon)
        angles = np.linspace(0, 2 * np.pi, 9)[:-1]
        polygon = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles),
        ])
        return polygon

    def _get_state(self, agent_id: str, agents_by_id: dict[str, Agent],
                   ego: VehicleState) -> Optional[VehicleState]:
        if agent_id == "ego" or agent_id == getattr(ego, 'id', None):
            return ego
        agent = agents_by_id.get(agent_id)
        return agent.state if agent else None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

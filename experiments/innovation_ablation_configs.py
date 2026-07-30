"""Innovation-point ablation configs for Research Point 3.

The four switches map paper-level claims to executable variants:

V: visibility/uncertainty-aware risk object modeling and adaptive margin
D: decoupled parallel waypoint correction and risk detection
M: multi-direction / multi-agent risk-aware avoidance
H: longer-horizon feasibility checking across future waypoints/actions
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from coop_safety.interface import Agent, PerceptionResult
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.method_variants import estimate_directional_ttc, estimate_rear_gap


@dataclass(frozen=True)
class InnovationFlags:
    use_visibility: bool = False
    use_decoupling: bool = False
    use_multidirection: bool = False
    use_long_horizon: bool = False

    @property
    def suffix(self) -> str:
        parts = []
        if self.use_visibility:
            parts.append("V")
        if self.use_decoupling:
            parts.append("D")
        if self.use_multidirection:
            parts.append("M")
        if self.use_long_horizon:
            parts.append("H")
        return "+".join(parts) if parts else "NONE"


class InnovationHybridBase(HybridSafetyConstraint):
    """Hybrid base with feature switches for ablation."""

    def __init__(self, *args, flags: InnovationFlags, fixed_margin: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.flags = flags
        self.fixed_margin = fixed_margin

    def _without_visibility(self, perception: PerceptionResult) -> PerceptionResult:
        if self.flags.use_visibility:
            return perception
        agents = [
            Agent(
                state=a.state,
                agent_type=a.agent_type,
                confidence=a.confidence,
                is_visible=True,
                source=a.source,
            )
            for a in perception.agents
        ]
        return PerceptionResult(
            timestamp=perception.timestamp,
            ego=perception.ego,
            agents=agents,
            lanes=perception.lanes,
            blind_spots=perception.blind_spots,
            visibility_map=perception.visibility_map,
        )

    def _get_safety_margin(self, agent, ego_speed: float) -> float:
        if self.flags.use_visibility:
            return super()._get_safety_margin(agent, ego_speed)

        s = agent.state
        margin = self.fixed_margin
        dist = math.sqrt(s.x**2 + s.y**2)
        if dist > 0.01:
            rel_vx = s.vx - ego_speed
            rel_vy = s.vy
            approach_speed = -(s.x * rel_vx + s.y * rel_vy) / dist
            if approach_speed > 0:
                margin *= (1.0 + self.approach_speed_factor * min(approach_speed / self.v_max, 1.0))
        margin += max(s.length, s.width) * 0.3
        return margin

    def _find_threats(self, wp: np.ndarray, agents, t_idx: int, ego_speed: float) -> list:
        if not self.flags.use_long_horizon and t_idx >= 3:
            return []
        threats = super()._find_threats(wp, agents, t_idx, ego_speed)
        if self.flags.use_multidirection or len(threats) <= 1:
            return threats
        return [min(threats, key=lambda item: item[3])]

    def _multi_agent_repulsion(self, wp: np.ndarray, threats: list, wp_risk: float = 0.0) -> np.ndarray:
        if self.flags.use_multidirection:
            return super()._multi_agent_repulsion(wp, threats, wp_risk=wp_risk)
        if not threats:
            return wp.copy()
        agent, ax, ay, dist, margin = min(threats, key=lambda item: item[3])
        dx = wp[0] - ax
        dy = wp[1] - ay
        d = max(dist, 0.1)
        push_dist = max(margin - dist + self.push_clearance, 1.0)
        new_wp = wp.copy()
        new_wp[0] += (dx / d) * push_dist
        new_wp[1] += (dy / d) * push_dist
        return new_wp

    def constrain_waypoints(self, waypoints: np.ndarray, perception: PerceptionResult) -> tuple:
        return super().constrain_waypoints(waypoints, self._without_visibility(perception))


class InnovationAblationMethod:
    """Executable combination of the four innovation switches."""

    def __init__(self, base: InnovationHybridBase, flags: InnovationFlags, ttc_override: float = 3.0,
                 rear_gap_guard: float = 18.0):
        self.base = base
        self.flags = flags
        self.ttc_override = ttc_override
        self.rear_gap_guard = rear_gap_guard
        self.name = f"Innov-{flags.suffix}"

    def constrain_waypoints(self, waypoints, perception):
        modified, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get("collision_prob", 0.0)
        n_geom = stats.get("n_geometric_threats", 0)
        min_ttc = stats.get("min_ttc", 999.0)

        if self.flags.use_multidirection:
            front_ttc, rear_ttc = estimate_directional_ttc(perception)
            rear_gap = estimate_rear_gap(perception)
        else:
            front_ttc, rear_ttc = min_ttc, 999.0
            rear_gap = 999.0

        if self.flags.use_decoupling:
            fire = ((prob > self.base.detection_threshold) and (n_geom > 0)) or (front_ttc < self.ttc_override)
        else:
            fire = (prob > self.base.detection_threshold) or (n_geom > 0) or (min_ttc < self.ttc_override)
            if not fire:
                modified = waypoints.copy()
                stats["modification_rate"] = 0.0
                stats["n_geometric_threats"] = 0

        stats["method"] = self.name
        stats["n_collisions_detected"] = 1 if fire else 0
        stats["front_side_ttc"] = front_ttc
        stats["rear_ttc"] = rear_ttc
        stats["rear_gap"] = rear_gap
        stats["innovation_visibility"] = int(self.flags.use_visibility)
        stats["innovation_decoupling"] = int(self.flags.use_decoupling)
        stats["innovation_multidirection"] = int(self.flags.use_multidirection)
        stats["innovation_long_horizon"] = int(self.flags.use_long_horizon)

        if self.flags.use_multidirection:
            close_rear = rear_gap < self.rear_gap_guard or rear_ttc < 2.5
            if close_rear and front_ttc >= 1.0:
                stats["lane_escape" if self.flags.use_long_horizon else "lane_escape_unchecked"] = 1
                factor = 1.0
            elif front_ttc < 1.0:
                if close_rear:
                    stats["lane_escape" if self.flags.use_long_horizon else "lane_escape_unchecked"] = 1
                factor = 0.5
            elif front_ttc < self.ttc_override:
                factor = 0.7
            elif fire:
                factor = 0.8
            else:
                factor = 1.0
        else:
            if min_ttc < 2.0:
                factor = 0.1
            elif min_ttc < self.ttc_override:
                factor = 0.3
            elif fire:
                factor = 0.8
            else:
                factor = 1.0

        stats["target_speed_factor"] = factor
        return modified, stats


def innovation_ablation_configs(v1):
    """Return no-point, single-point, pairwise, and full 4-point variants."""
    base_kwargs = dict(
        detector_model=v1,
        base_margin_visible=2.5,
        base_margin_invisible=4.0,
        detection_threshold=0.30,
    )
    combos = [
        InnovationFlags(False, False, False, False),
        InnovationFlags(True, False, False, False),
        InnovationFlags(False, True, False, False),
        InnovationFlags(False, False, True, False),
        InnovationFlags(False, False, False, True),
        InnovationFlags(True, True, False, False),
        InnovationFlags(True, False, True, False),
        InnovationFlags(True, False, False, True),
        InnovationFlags(False, True, True, False),
        InnovationFlags(False, True, False, True),
        InnovationFlags(False, False, True, True),
        InnovationFlags(True, True, True, True),
    ]

    configs = []
    for flags in combos:
        name = f"Innov-{flags.suffix}"

        def factory(flags=flags):
            base = InnovationHybridBase(flags=flags, **base_kwargs)
            return InnovationAblationMethod(base, flags)

        configs.append((name, factory, True))
    return configs

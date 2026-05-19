from __future__ import annotations
"""Minimum harm mode for unavoidable collision scenarios.

When no safe feasible region exists, this module selects the action that
minimizes expected harm. Corresponds to research plan step 3.6.

Innovation Point 3: Brings safety failure under control.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, Point

from ..interface import (
    VehicleState, CollisionEvent, SafeActionSpace, ConstraintMode,
)
from ..perception.dynamics import BicycleModel, get_dynamics_params


class MinimumHarmPlanner:
    """Select minimum-harm action when collision is unavoidable.

    Triggered when:
    1. No feasible region remains after hierarchical tightening (step 3.4)
    2. All feasible actions fail long-term feasibility check (step 3.5)

    Strategy:
    - Evaluate all possible collision events
    - For each candidate action, estimate which collision would occur
    - Select the action that minimizes: severity × probability
    - Prefer collisions with lower severity (property damage over injury)

    Inspired by aviation "controlled safe failure" concepts (TCAS RA logic).
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.action_samples = cfg.get("action_samples", 12)  # Reduced from 20
        self.sim_horizon = cfg.get("sim_horizon", 1.5)  # Reduced from 2.0
        self.sim_dt = cfg.get("sim_dt", 0.2)  # Coarser from 0.1
        # Severity weights for optimization
        self.pedestrian_penalty = cfg.get("pedestrian_penalty", 5.0)
        self.head_on_penalty = cfg.get("head_on_penalty", 3.0)

    def plan(self, ego: VehicleState,
             events: list[CollisionEvent],
             original_feasible: Optional[Polygon] = None,
             ) -> SafeActionSpace:
        """Plan minimum-harm action.

        Args:
            ego: Current ego vehicle state
            events: All potential collision events
            original_feasible: The feasible region before it became empty (for reference)

        Returns:
            SafeActionSpace in MINIMUM_HARM mode
        """
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        ego_state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        # Sample candidate actions (bias toward braking and evasive maneuvers)
        candidates = self._generate_candidates(params)

        best_action = None
        best_cost = float('inf')
        best_event = None
        reasoning = []

        for action in candidates:
            # Simulate ego forward
            trajectory = model.predict_trajectory(
                ego_state, action, self.sim_horizon, self.sim_dt
            )

            # Evaluate cost: expected harm from each collision event
            cost = self._evaluate_harm(trajectory, events, ego)

            if cost < best_cost:
                best_cost = cost
                best_action = action
                # Find the dominant collision event for this action
                best_event = self._find_dominant_event(trajectory, events)

        # Build minimum-harm safe action space
        # Use a small feasible region around the best action's trajectory
        if best_action is not None:
            traj = model.predict_trajectory(ego_state, best_action, self.sim_horizon, self.sim_dt)
            # Feasible region: narrow corridor along the minimum-harm trajectory
            feasible_poly = self._trajectory_to_corridor(traj[:, :2], width=2.0)
        else:
            # Absolute worst case: emergency brake
            best_action = np.array([-params.max_deceleration, 0.0])
            feasible_poly = Point(ego.x, ego.y).buffer(5.0)
            reasoning.append("Emergency brake: no better option found")

        reasoning.insert(0, f"MINIMUM HARM MODE: cost={best_cost:.3f}")
        if best_event:
            reasoning.append(
                f"Expected collision: {best_event.collision_type}, "
                f"severity={best_event.severity:.2f}, prob={best_event.probability:.2f}"
            )

        return SafeActionSpace(
            feasible_region=np.array(feasible_poly.exterior.coords)
            if hasattr(feasible_poly, 'exterior') else np.array([[ego.x, ego.y]]),
            max_acceleration=best_action[0] if best_action[0] > 0 else 0.0,
            min_acceleration=best_action[0] if best_action[0] < 0 else -params.max_deceleration,
            max_steering=abs(best_action[1]),
            max_speed=max(ego.velocity + best_action[0] * self.sim_horizon, 0),
            mode=ConstraintMode.MINIMUM_HARM,
            safety_margin_ttc=0.0,
            feasibility_horizon=0.0,
            reasoning=reasoning,
            risk_sources=[e.event_id for e in events[:3]],
            future_feasible=False,
            future_feasible_horizon=0.0,
            min_harm_target=best_event,
            min_harm_action=best_action,
        )

    def _generate_candidates(self, params) -> list[np.ndarray]:
        """Generate candidate actions biased toward safety."""
        actions = []

        # Emergency brake (highest priority)
        actions.append(np.array([-params.max_deceleration, 0.0]))

        # Hard braking with steering (5 options)
        for steer in np.linspace(-params.max_steering_angle, params.max_steering_angle, 5):
            actions.append(np.array([-params.max_deceleration * 0.8, steer]))

        # Moderate braking with steering (3 options)
        for steer in np.linspace(-params.max_steering_angle * 0.5,
                                 params.max_steering_angle * 0.5, 3):
            actions.append(np.array([-params.max_deceleration * 0.5, steer]))

        return actions

    def _evaluate_harm(self, trajectory: np.ndarray,
                       events: list[CollisionEvent],
                       ego: VehicleState) -> float:
        """Evaluate expected harm for a trajectory across all events.

        Cost = Σ (severity × probability × proximity_factor × type_penalty)
        """
        total_cost = 0.0

        for event in events:
            # Check proximity of trajectory to event region
            if len(event.spatial_region) < 3:
                continue

            event_center = np.mean(event.spatial_region, axis=0)

            # Minimum distance from trajectory to event center
            dists = np.linalg.norm(trajectory[:, :2] - event_center, axis=1)
            min_dist = np.min(dists)

            # Proximity factor: 1.0 at center, decaying with distance
            proximity = np.exp(-min_dist / 5.0)

            # Type penalty
            type_penalty = 1.0
            if event.collision_type == "pedestrian":
                type_penalty = self.pedestrian_penalty
            elif event.collision_type == "head_on":
                type_penalty = self.head_on_penalty

            cost = event.severity * event.probability * proximity * type_penalty
            total_cost += cost

        # State is [x, y, heading, velocity]; penalize impact speed, not heading.
        final_speed = abs(float(trajectory[-1, 3])) if trajectory.shape[1] > 3 else ego.velocity
        total_cost += final_speed * 0.01

        return total_cost

    def _find_dominant_event(self, trajectory: np.ndarray,
                             events: list[CollisionEvent]) -> Optional[CollisionEvent]:
        """Find the collision event most likely to occur for this trajectory."""
        best_event = None
        best_score = 0.0

        for event in events:
            if len(event.spatial_region) < 3:
                continue
            event_center = np.mean(event.spatial_region, axis=0)
            dists = np.linalg.norm(trajectory[:, :2] - event_center, axis=1)
            proximity = np.exp(-np.min(dists) / 5.0)
            score = event.probability * proximity
            if score > best_score:
                best_score = score
                best_event = event

        return best_event

    @staticmethod
    def _trajectory_to_corridor(positions: np.ndarray, width: float) -> Polygon:
        """Convert a trajectory to a corridor polygon."""
        from shapely.geometry import LineString
        if len(positions) < 2:
            return Point(positions[0]).buffer(width)
        line = LineString(positions)
        return line.buffer(width)

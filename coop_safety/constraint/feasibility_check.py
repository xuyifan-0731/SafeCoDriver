from __future__ import annotations
"""Long-term feasibility check.

Verifies that the current safe action space remains feasible for the next
N seconds, preventing "dead-end" constraints where the vehicle is safe now
but has no viable future actions. Corresponds to research plan step 3.5.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, Point

from ..interface import VehicleState, Agent, PerceptionResult
from ..perception.dynamics import BicycleModel, get_dynamics_params
from ..perception.prediction import predict_all_agents


class FeasibilityChecker:
    """Check long-term feasibility of safety constraints.

    For each candidate safe action, simulate forward and verify that
    viable safe regions still exist at future timesteps.

    Innovation Point 2: Prevents feasibility rupture.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.check_horizon = cfg.get("check_horizon", 3.0)   # How far to look ahead
        self.check_interval = cfg.get("check_interval", 1.0)  # Check every N seconds (was 0.5)
        self.min_future_area = cfg.get("min_future_area", 10.0)  # m² minimum feasible area
        self.n_action_samples = cfg.get("n_action_samples", 4)  # Reduced from 8

    def check(self, feasible_region: Polygon,
              ego: VehicleState,
              agents: list[Agent],
              ) -> tuple[bool, float, list[str]]:
        """Check if current feasible region allows long-term viability.

        Simulates ego vehicle taking actions within the feasible region
        and checks if future feasible regions remain non-empty.

        Args:
            feasible_region: Current constrained feasible region
            ego: Ego vehicle state
            agents: Current agents (for predicting future environment)

        Returns:
            (is_feasible, feasible_horizon_achieved, reasoning)
        """
        if feasible_region.is_empty or feasible_region.area < self.min_future_area:
            return False, 0.0, ["Current feasible region too small"]

        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        ego_state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        # Sample actions within feasible region
        feasible_actions = self._sample_feasible_actions(
            ego_state, params, feasible_region
        )

        if not feasible_actions:
            return False, 0.0, ["No feasible actions found within constraint"]

        # For each sampled action, simulate forward and check future feasibility
        best_horizon = 0.0
        best_reasoning = []

        for action in feasible_actions:
            horizon_achieved, reasons = self._simulate_forward(
                model, ego_state, action, agents, feasible_region
            )
            if horizon_achieved > best_horizon:
                best_horizon = horizon_achieved
                best_reasoning = reasons

        is_feasible = best_horizon >= self.check_horizon * 0.8  # Allow 80% threshold
        return is_feasible, best_horizon, best_reasoning

    def _simulate_forward(self, model: BicycleModel,
                          ego_state: np.ndarray,
                          action: np.ndarray,
                          agents: list[Agent],
                          current_feasible: Polygon,
                          ) -> tuple[float, list[str]]:
        """Simulate one action forward and check feasibility at each checkpoint.

        Returns:
            (max_feasible_horizon, reasoning_list)
        """
        reasoning = []
        n_checks = int(self.check_horizon / self.check_interval)
        state = ego_state.copy()

        # Predict agent positions over time
        agent_predictions = predict_all_agents(agents, self.check_horizon, 0.1)

        for check_idx in range(n_checks):
            t = (check_idx + 1) * self.check_interval
            steps = int(self.check_interval / 0.1)

            # Propagate ego state
            for _ in range(steps):
                state = model.step(state, action, 0.1)

            ego_point = Point(state[0], state[1])

            # Check 1: Is ego still within a reasonable envelope?
            # We use a distance-based check rather than strict containment,
            # because the feasible region is defined at t=0 but ego moves.
            # Instead check if ego is reasonably close to the feasible boundary.
            dist_to_feasible = ego_point.distance(current_feasible.boundary)
            if not current_feasible.contains(ego_point) and dist_to_feasible > 20.0:
                reasoning.append(f"t={t:.1f}s: ego far from feasible region ({dist_to_feasible:.1f}m)")
                return t - self.check_interval, reasoning

            # Check 2: Do agents at time t leave enough space?
            # Simplified: check if ego point is far enough from predicted agent positions
            min_agent_dist = float('inf')
            for agent_id, traj in agent_predictions.items():
                t_idx = min(int(t / 0.1), len(traj) - 1)
                if t_idx >= 0 and t_idx < len(traj):
                    dist = np.linalg.norm(state[:2] - traj[t_idx])
                    min_agent_dist = min(min_agent_dist, dist)

            if min_agent_dist < 3.0:  # Too close to any predicted agent
                reasoning.append(
                    f"t={t:.1f}s: predicted agent within {min_agent_dist:.1f}m"
                )
                return t - self.check_interval, reasoning

        reasoning.append(f"Feasible through full horizon ({self.check_horizon:.1f}s)")
        return self.check_horizon, reasoning

    def _sample_feasible_actions(self, ego_state: np.ndarray,
                                 params, feasible: Polygon) -> list[np.ndarray]:
        """Sample control actions that lead to positions within the feasible region."""
        actions = []
        # Sparse sampling: 4 accelerations × 4 steerings = 16 candidates
        accels = np.array([-params.max_deceleration * 0.3, 0,
                           params.max_acceleration * 0.3, params.max_acceleration * 0.5])
        steers = np.array([-params.max_steering_angle * 0.3, 0,
                           params.max_steering_angle * 0.3])

        model = BicycleModel(params)

        for a in accels:
            for s in steers:
                action = np.array([a, s])
                state = ego_state.copy()
                # Quick 1-second forward check (5 steps of 0.2s)
                for _ in range(5):
                    state = model.step(state, action, 0.2)
                if feasible.contains(Point(state[0], state[1])):
                    actions.append(action)
                    if len(actions) >= 3:  # Early exit: 3 feasible actions is enough
                        return actions

        return actions

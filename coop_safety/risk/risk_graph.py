from __future__ import annotations
"""RiskGraph: Pairwise interaction conflict analysis.

Builds a graph where nodes are traffic participants and edges represent
potential conflicts with TTC and collision probability.
Corresponds to research plan step 2.5.
"""

import numpy as np
from typing import Optional
import itertools

from ..interface import Agent, VehicleState, ConflictEdge
from ..perception.prediction import predict_agent
from ..utils.metrics import compute_ttc


class RiskGraphBuilder:
    """Build pairwise conflict graph from agents and their predictions.

    RiskGraph focuses on agent-level interactions (unlike RiskMap which is spatial).
    Each edge contains:
    - TTC (Time-to-Collision)
    - Collision probability
    - Predicted collision point and time window
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.prediction_horizon = cfg.get("prediction_horizon", 5.0)
        self.prediction_dt = cfg.get("prediction_dt", 0.1)
        self.collision_radius = cfg.get("collision_radius", 3.0)  # meters
        self.max_ttc = cfg.get("max_ttc", 8.0)  # Only consider conflicts within this TTC

    def build(self, ego: VehicleState, agents: list[Agent]) -> list[ConflictEdge]:
        """Build conflict graph for ego and all agents.

        Analyzes pairwise interactions between ego and each agent,
        and between agents that may affect ego's path.

        Args:
            ego: Ego vehicle state
            agents: All detected agents

        Returns:
            List of ConflictEdge representing potential conflicts
        """
        edges = []

        # Create ego agent wrapper
        ego_agent = Agent(state=ego)

        # Predict ego trajectory (constant velocity as default)
        ego_traj = predict_agent(ego_agent, self.prediction_horizon, self.prediction_dt)

        # Predict all agent trajectories
        agent_trajs = {}
        for agent in agents:
            agent_trajs[agent.state.id] = predict_agent(
                agent, self.prediction_horizon, self.prediction_dt
            )

        # Ego vs each agent
        for agent in agents:
            edge = self._analyze_pair(
                ego.id if hasattr(ego, 'id') else "ego",
                ego, ego_traj,
                agent.state.id, agent.state, agent_trajs[agent.state.id],
            )
            if edge is not None:
                edges.append(edge)

        # Agent vs agent (only for nearby agents that may affect ego)
        nearby_agents = [a for a in agents
                         if self._distance(ego, a.state) < self.prediction_horizon * 30]
        for a1, a2 in itertools.combinations(nearby_agents, 2):
            edge = self._analyze_pair(
                a1.state.id, a1.state, agent_trajs[a1.state.id],
                a2.state.id, a2.state, agent_trajs[a2.state.id],
            )
            if edge is not None:
                edges.append(edge)

        return edges

    def _analyze_pair(self, id_a: str, state_a: VehicleState, traj_a: np.ndarray,
                      id_b: str, state_b: VehicleState, traj_b: np.ndarray,
                      ) -> Optional[ConflictEdge]:
        """Analyze conflict potential between two agents.

        Compares predicted trajectories point-by-point to find the closest
        approach distance and time.
        """
        # Instantaneous TTC
        ttc = compute_ttc(
            np.array([state_a.x, state_a.y]),
            np.array([state_a.vx, state_a.vy]),
            np.array([state_b.x, state_b.y]),
            np.array([state_b.vx, state_b.vy]),
        )

        if ttc > self.max_ttc:
            return None  # No near-term conflict

        # Skip if agents are well separated and moving similarly
        dist = np.linalg.norm(np.array([state_a.x - state_b.x, state_a.y - state_b.y]))
        if dist > 30.0 and ttc > 5.0:
            return None

        # Trajectory-based analysis: find minimum distance over time
        min_len = min(len(traj_a), len(traj_b))
        if min_len == 0:
            return None

        distances = np.linalg.norm(traj_a[:min_len] - traj_b[:min_len], axis=1)
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        # Collision radius considers vehicle sizes
        effective_radius = self.collision_radius + (state_a.width + state_b.width) / 2

        if min_dist > effective_radius * 3:
            return None  # Trajectories are well separated

        # Collision probability: sigmoid based on minimum distance
        # P = 1 / (1 + exp(k * (d - threshold)))
        k = 2.0
        threshold = effective_radius
        collision_prob = 1.0 / (1.0 + np.exp(k * (min_dist - threshold)))

        # Find conflict time window (when distance < 2 * effective_radius)
        conflict_mask = distances < 2 * effective_radius
        if conflict_mask.any():
            conflict_indices = np.where(conflict_mask)[0]
            t_start = conflict_indices[0] * self.prediction_dt
            t_end = conflict_indices[-1] * self.prediction_dt
        else:
            t_start = min_dist_idx * self.prediction_dt
            t_end = t_start

        # Collision point
        collision_point = (traj_a[min_dist_idx] + traj_b[min_dist_idx]) / 2

        return ConflictEdge(
            agent_a_id=id_a,
            agent_b_id=id_b,
            ttc=ttc,
            collision_probability=collision_prob,
            collision_point=collision_point,
            time_window=(t_start, t_end),
        )

    @staticmethod
    def _distance(a: VehicleState, b: VehicleState) -> float:
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

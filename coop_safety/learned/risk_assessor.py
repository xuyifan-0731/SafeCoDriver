"""Learned risk assessor — drop-in replacement for rule-based RiskMap + RiskGraph.

Uses the trained RiskAssessmentNetwork to predict:
  1. Spatial risk at query points (replaces RiskMap)
  2. Pairwise collision probability and TTC (replaces RiskGraph)

The RiskEvents layer still uses rule-based logic on top of learned predictions.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, RiskRegion, RiskLevel, ConflictEdge,
)
from coop_safety.learned.risk_network import RiskAssessmentNetwork


class LearnedRiskAssessor:
    """Use trained network for risk assessment instead of rule-based system."""

    def __init__(self, model_path: str = None, device: str = "cuda:0"):
        if model_path is None:
            model_path = "/raid/xuyifan/jiqiuyu/models/risk_net_best.pt"

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = RiskAssessmentNetwork(agent_feat_dim=8, embed_dim=32).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.ego_mean = checkpoint.get("ego_mean", np.zeros(8))
        self.ego_std = checkpoint.get("ego_std", np.ones(8))
        self.agent_mean = checkpoint.get("agent_mean", np.zeros(8))
        self.agent_std = checkpoint.get("agent_std", np.ones(8))

        print(f"[LearnedRiskAssessor] Loaded from {model_path}")

    def _encode_type(self, t: str) -> float:
        return {"car": 0, "truck": 1, "bus": 2, "pedestrian": 3, "bicycle": 4, "motorcycle": 5}.get(t, 0)

    def assess_risk_map(self, perception: PerceptionResult,
                        grid_range: float = 40.0, cell_size: float = 8.0) -> list[RiskRegion]:
        """Predict spatial risk using learned network (replaces RiskMapBuilder)."""
        ego = perception.ego
        agents = perception.agents

        # Encode agents
        max_agents = 40
        agent_feats = np.zeros((1, max_agents, 8), dtype=np.float32)
        mask = np.zeros((1, max_agents), dtype=bool)
        for i, a in enumerate(agents[:max_agents]):
            s = a.state
            agent_feats[0, i] = [s.x, s.y, s.vx, s.vy, s.heading, s.length, s.width, self._encode_type(s.vehicle_type)]
            mask[0, i] = True

        ego_feat = np.array([[ego.x, ego.y, ego.vx, ego.vy, ego.heading, ego.velocity, 4.5, 0]], dtype=np.float32)

        # Normalize
        agent_feats_norm = (agent_feats - self.agent_mean) / self.agent_std
        ego_feat_norm = (ego_feat - self.ego_mean) / self.ego_std

        # Generate grid query points (ego-relative)
        n_cells_x = int(2 * grid_range / cell_size)
        n_cells_y = int(2 * grid_range / cell_size)
        query_pts = []
        cell_corners = []
        cos_h, sin_h = np.cos(ego.heading), np.sin(ego.heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        for ix in range(n_cells_x):
            for iy in range(n_cells_y):
                cx = -grid_range + (ix + 0.5) * cell_size
                cy = -grid_range + (iy + 0.5) * cell_size
                query_pts.append([cx, cy])

                # Cell corners in global frame
                corners = np.array([
                    [cx - cell_size/2, cy - cell_size/2],
                    [cx + cell_size/2, cy - cell_size/2],
                    [cx + cell_size/2, cy + cell_size/2],
                    [cx - cell_size/2, cy + cell_size/2],
                ])
                global_corners = corners @ rot.T + np.array([ego.x, ego.y])
                cell_corners.append(global_corners)

        query_pts = np.array(query_pts, dtype=np.float32).reshape(1, -1, 2)

        # Predict
        with torch.no_grad():
            out = self.model(
                torch.FloatTensor(ego_feat_norm).to(self.device),
                torch.FloatTensor(agent_feats_norm).to(self.device),
                torch.BoolTensor(mask).to(self.device),
                query_points=torch.FloatTensor(query_pts).to(self.device),
            )
            risk_scores = out["risk"][0, :, 0].cpu().numpy()  # (Q,)

        # Convert to RiskRegions
        regions = []
        for qi, (score, corners) in enumerate(zip(risk_scores, cell_corners)):
            if score >= 0.35:
                level = RiskLevel.HIGH
            elif score >= 0.15:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            regions.append(RiskRegion(
                polygon=corners,
                risk_level=level,
                risk_score=float(score),
            ))

        return regions

    def assess_conflicts(self, ego: VehicleState, agents: list[Agent]) -> list[ConflictEdge]:
        """Predict pairwise conflicts using learned network (replaces RiskGraphBuilder)."""
        max_agents = 40
        agent_feats = np.zeros((1, max_agents, 8), dtype=np.float32)
        mask = np.zeros((1, max_agents), dtype=bool)
        for i, a in enumerate(agents[:max_agents]):
            s = a.state
            agent_feats[0, i] = [s.x, s.y, s.vx, s.vy, s.heading, s.length, s.width, self._encode_type(s.vehicle_type)]
            mask[0, i] = True

        ego_feat = np.array([[ego.x, ego.y, ego.vx, ego.vy, ego.heading, ego.velocity, 4.5, 0]], dtype=np.float32)

        n_valid = min(len(agents), max_agents)
        if n_valid < 2:
            return []

        # Generate pair indices (ego vs each agent + agent-agent for nearby)
        pairs = []
        pair_ids = []
        for i in range(n_valid):
            pairs.append([0, i])  # ego vs agent_i (using index 0 as proxy)
            pair_ids.append(("ego", agents[i].state.id))
        for i in range(min(n_valid, 10)):
            for j in range(i + 1, min(n_valid, 10)):
                pairs.append([i, j])
                pair_ids.append((agents[i].state.id, agents[j].state.id))

        if not pairs:
            return []

        pair_indices = np.array(pairs, dtype=np.float32).reshape(1, -1, 2)

        # Normalize
        agent_feats_norm = (agent_feats - self.agent_mean) / self.agent_std

        with torch.no_grad():
            out = self.model(
                torch.FloatTensor((ego_feat - self.ego_mean) / self.ego_std).to(self.device),
                torch.FloatTensor(agent_feats_norm).to(self.device),
                torch.BoolTensor(mask).to(self.device),
                pair_indices=torch.FloatTensor(pair_indices).to(self.device),
            )
            preds = out["conflict"][0].cpu().numpy()  # (P, 2): prob, ttc

        edges = []
        for pi, ((id_a, id_b), pred) in enumerate(zip(pair_ids, preds)):
            prob = float(pred[0])
            ttc = float(pred[1])
            if prob < 0.05 and ttc > 10:
                continue  # Skip negligible conflicts
            edges.append(ConflictEdge(
                agent_a_id=id_a,
                agent_b_id=id_b,
                ttc=ttc,
                collision_probability=prob,
            ))

        return edges

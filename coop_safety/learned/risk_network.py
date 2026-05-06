"""Risk Assessment Network — learned replacement for RiskMap + RiskGraph.

Architecture:
  Agent Encoder: per-agent MLP(8→64→32) + max-pool → scene vector (32-dim)
  Risk Head: MLP(32+2→64→32→1) → per-point risk score
  Conflict Head: MLP(32+32→64→32→2) → per-pair (collision_prob, ttc)

Total parameters: ~0.2M
Inference: <2ms per frame on GPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AgentEncoder(nn.Module):
    """Encode each agent's features, then aggregate into a scene vector."""

    def __init__(self, in_dim=8, hidden=64, out_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, agents: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            agents: (B, N, in_dim) agent features
            mask: (B, N) boolean mask (True = valid agent)
        Returns:
            agent_embeds: (B, N, out_dim) per-agent embeddings
            scene_vec: (B, out_dim) aggregated scene vector
        """
        embeds = self.mlp(agents)  # (B, N, out_dim)
        # Masked max-pool
        embeds_masked = embeds.clone()
        embeds_masked[~mask] = -1e9
        scene_vec = embeds_masked.max(dim=1)[0]  # (B, out_dim)
        return embeds, scene_vec


class RiskHead(nn.Module):
    """Predict risk score at a query point given scene context."""

    def __init__(self, scene_dim=32, point_dim=2, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(scene_dim + point_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, scene_vec: torch.Tensor, query_points: torch.Tensor):
        """
        Args:
            scene_vec: (B, scene_dim)
            query_points: (B, Q, 2) query (x, y) in ego-relative frame
        Returns:
            risk: (B, Q, 1) risk scores [0, 1]
        """
        B, Q, _ = query_points.shape
        scene_expanded = scene_vec.unsqueeze(1).expand(B, Q, -1)  # (B, Q, scene_dim)
        x = torch.cat([scene_expanded, query_points], dim=-1)  # (B, Q, scene_dim+2)
        return self.mlp(x)


class ConflictHead(nn.Module):
    """Predict collision probability and TTC for agent pairs."""

    def __init__(self, embed_dim=32, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # (collision_prob, ttc_estimate)
        )

    def forward(self, embed_i: torch.Tensor, embed_j: torch.Tensor):
        """
        Args:
            embed_i, embed_j: (B, P, embed_dim) embeddings of agent pairs
        Returns:
            out: (B, P, 2) — [:,:,0]=collision_prob (sigmoid), [:,:,1]=ttc (relu)
        """
        x = torch.cat([embed_i, embed_j], dim=-1)
        out = self.mlp(x)
        out[:, :, 0] = torch.sigmoid(out[:, :, 0])  # prob in [0, 1]
        out[:, :, 1] = F.relu(out[:, :, 1])  # ttc >= 0
        return out


class RiskAssessmentNetwork(nn.Module):
    """Complete Risk Assessment Network combining all heads."""

    def __init__(self, agent_feat_dim=8, embed_dim=32):
        super().__init__()
        self.encoder = AgentEncoder(agent_feat_dim, 64, embed_dim)
        self.risk_head = RiskHead(embed_dim, 2, 64)
        self.conflict_head = ConflictHead(embed_dim, 64)

    def forward(self, ego: torch.Tensor, agents: torch.Tensor, mask: torch.Tensor,
                query_points: torch.Tensor = None,
                pair_indices: torch.Tensor = None):
        """
        Args:
            ego: (B, feat_dim) ego features
            agents: (B, N, feat_dim) agent features
            mask: (B, N) valid agent mask
            query_points: (B, Q, 2) optional spatial query points (ego-relative)
            pair_indices: (B, P, 2) optional pair indices for conflict prediction
        Returns:
            dict with 'risk' and/or 'conflict' predictions
        """
        agent_embeds, scene_vec = self.encoder(agents, mask)

        result = {"scene_vec": scene_vec}

        if query_points is not None:
            risk = self.risk_head(scene_vec, query_points)
            result["risk"] = risk  # (B, Q, 1)

        if pair_indices is not None:
            B, P, _ = pair_indices.shape
            # Gather pair embeddings
            idx_i = pair_indices[:, :, 0].long().clamp(0, agents.shape[1] - 1)
            idx_j = pair_indices[:, :, 1].long().clamp(0, agents.shape[1] - 1)
            embed_i = torch.gather(agent_embeds, 1, idx_i.unsqueeze(-1).expand(-1, -1, agent_embeds.shape[-1]))
            embed_j = torch.gather(agent_embeds, 1, idx_j.unsqueeze(-1).expand(-1, -1, agent_embeds.shape[-1]))
            conflict = self.conflict_head(embed_i, embed_j)
            result["conflict"] = conflict  # (B, P, 2)

        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

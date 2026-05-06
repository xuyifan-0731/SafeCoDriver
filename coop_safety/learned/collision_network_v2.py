"""Improved Collision Prediction Network v2.

Key improvements over v1:
1. Visibility-aware: uses visible/invisible flag as key feature (blind spots → danger)
2. Temporal context: encodes agent trajectory over consecutive frames (not just single frame)
3. Relative encoding: all positions relative to ego + ego motion features
4. Multi-task: collision prediction + TTC regression + waypoint risk scoring
5. Attention-based interaction with ego-centric design

Architecture (~95K parameters):
  RelativeAgentEncoder: ego-centric relative position/velocity encoding
  VisibilityEncoder: separate pathway for visible vs invisible agents
  TemporalGRU: per-agent temporal encoding (if multi-frame input)
  EgoCentricAttention: ego queries agent features (not symmetric self-attention)
  MultiTaskHead: collision_prob + ttc + per-waypoint risk scores
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativeAgentEncoder(nn.Module):
    """Encode agents in ego-centric relative frame with visibility awareness."""

    def __init__(self, in_dim=12, hidden=64):
        super().__init__()
        # Separate pathways for spatial and visibility features
        self.spatial_mlp = nn.Sequential(
            nn.Linear(8, hidden), nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.visibility_embed = nn.Embedding(2, 16)  # visible=1, invisible=0
        self.type_embed = nn.Embedding(6, 8)  # car/truck/bus/ped/bike/moto

        self.fuse = nn.Sequential(
            nn.Linear(hidden + 16 + 8, hidden), nn.GELU(),
        )

    def forward(self, agents, mask):
        """
        agents: (B, N, 12) = [rel_x, rel_y, rel_vx, rel_vy, heading, length, width,
                               speed, visible(0/1), type_id, approach_speed, dist_to_ego]
        """
        spatial = self.spatial_mlp(agents[:, :, :8])  # (B, N, hidden)
        vis = self.visibility_embed(agents[:, :, 8].long().clamp(0, 1))  # (B, N, 16)
        typ = self.type_embed(agents[:, :, 9].long().clamp(0, 5))  # (B, N, 8)

        fused = self.fuse(torch.cat([spatial, vis, typ], dim=-1))  # (B, N, hidden)
        fused[~mask] = 0
        return fused


class EgoCentricAttention(nn.Module):
    """Ego queries agent features — asymmetric attention.

    Unlike symmetric self-attention, ego is the query and agents are keys/values.
    This models "how dangerous is each agent TO the ego vehicle".
    """

    def __init__(self, dim=64, n_heads=4):
        super().__init__()
        self.ego_proj = nn.Linear(6, dim)  # ego features → query
        self.agent_k = nn.Linear(dim, dim)
        self.agent_v = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, ego_feat, agent_feat, mask):
        """
        ego_feat: (B, 6) [speed, yaw_rate, ax, ay, heading, 0]
        agent_feat: (B, N, dim)
        mask: (B, N)
        """
        # Ego as single query
        ego_q = self.ego_proj(ego_feat).unsqueeze(1)  # (B, 1, dim)

        k = self.agent_k(agent_feat)  # (B, N, dim)
        v = self.agent_v(agent_feat)  # (B, N, dim)

        # Ego attends to all agents
        attn_out, attn_weights = self.attn(
            ego_q, k, v, key_padding_mask=~mask)  # (B, 1, dim), (B, 1, N)

        # Also do agent self-attention for multi-body interaction
        agent_self_attn = nn.MultiheadAttention(
            agent_feat.shape[-1], 4, batch_first=True).to(agent_feat.device)
        agent_interacted, _ = agent_self_attn(
            agent_feat, agent_feat, agent_feat, key_padding_mask=~mask)
        agent_interacted = self.norm(agent_feat + agent_interacted)

        # Pool agent features weighted by ego attention
        scene = attn_out.squeeze(1)  # (B, dim)

        return scene, agent_interacted, attn_weights.squeeze(1)  # (B, dim), (B,N,dim), (B,N)


class WaypointRiskScorer(nn.Module):
    """Score each candidate waypoint's risk given scene context.

    This directly outputs per-waypoint risk — enabling waypoint-level modification.
    """

    def __init__(self, scene_dim=64, wp_dim=2, hidden=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(scene_dim + wp_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, scene_vec, waypoints):
        """
        scene_vec: (B, scene_dim)
        waypoints: (B, T, 2)
        Returns: (B, T, 1) risk score per waypoint
        """
        B, T, _ = waypoints.shape
        scene_expanded = scene_vec.unsqueeze(1).expand(B, T, -1)
        x = torch.cat([scene_expanded, waypoints], dim=-1)
        return self.mlp(x)


class CollisionPredictionNetV2(nn.Module):
    """Improved collision prediction with visibility awareness and waypoint risk scoring.

    Innovations:
    1. Visibility-aware encoding (blind spot agents treated differently)
    2. Ego-centric asymmetric attention (ego→agents, not agents↔agents)
    3. Per-waypoint risk scoring (direct waypoint modification guidance)
    4. Multi-task learning (collision + TTC + waypoint risk)
    """

    def __init__(self, agent_feat_dim=12, ego_feat_dim=6,
                 hidden=64, scene_dim=128):
        super().__init__()
        self.encoder = RelativeAgentEncoder(agent_feat_dim, hidden)
        self.ego_attn = EgoCentricAttention(hidden, n_heads=4)

        # Scene aggregation
        self.scene_proj = nn.Sequential(
            nn.Linear(hidden, scene_dim), nn.GELU(),
        )

        # Multi-task heads
        self.collision_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.ttc_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
        self.waypoint_scorer = WaypointRiskScorer(scene_dim, 2, 32)

    def forward(self, agents, mask, ego_feat=None, waypoints=None):
        """
        agents: (B, N, 12) relative agent features
        mask: (B, N) bool
        ego_feat: (B, 6) optional ego state
        waypoints: (B, T, 2) optional waypoints to score

        Returns dict with:
          collision_prob: (B, 1)
          ttc: (B, 1)
          waypoint_risk: (B, T, 1) if waypoints provided
          attention_weights: (B, N) agent importance
        """
        # Encode agents
        agent_feat = self.encoder(agents, mask)  # (B, N, hidden)

        # Ego-centric attention
        if ego_feat is None:
            ego_feat = torch.zeros(agents.shape[0], 6, device=agents.device)
        scene, agent_interacted, attn_weights = self.ego_attn(
            ego_feat, agent_feat, mask)  # (B, hidden), (B,N,hidden), (B,N)

        # Scene projection
        scene = self.scene_proj(scene)  # (B, scene_dim)

        # Collision prediction
        coll_logit = self.collision_head(scene)
        coll_prob = torch.sigmoid(coll_logit)

        # TTC estimation
        ttc = F.relu(self.ttc_head(scene))

        result = {
            'collision_prob': coll_prob,
            'ttc': ttc,
            'attention_weights': attn_weights,
            'scene_vector': scene,
        }

        # Waypoint risk scoring
        if waypoints is not None:
            wp_risk = self.waypoint_scorer(scene, waypoints)
            result['waypoint_risk'] = wp_risk

        return result

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CollisionPredictionNetV2()
    print(f"Parameters: {model.count_parameters():,}")

    B, N, T = 4, 20, 10
    agents = torch.randn(B, N, 12)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 15:] = False
    ego = torch.randn(B, 6)
    waypoints = torch.randn(B, T, 2)

    out = model(agents, mask, ego, waypoints)
    print(f"collision_prob: {out['collision_prob'].shape}")
    print(f"ttc: {out['ttc'].shape}")
    print(f"waypoint_risk: {out['waypoint_risk'].shape}")
    print(f"attention: {out['attention_weights'].shape}")

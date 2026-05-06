"""Collision Prediction Network — trained on DeepAccident GT labels.

Predicts: P(collision within T frames) for the current scene.
Uses attention-based agent interaction modeling.

Architecture (~50K parameters):
  AgentEncoder: MLP(10→64→64) per agent
  Interaction: Multi-head Self-Attention (4 heads, dim=64)
  ScenePool: Attention-weighted pooling → 128-dim scene vector
  CollisionHead: MLP(128→64→1) → sigmoid → P(collision)
  TTCHead: MLP(128→64→1) → relu → estimated TTC
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class AgentEncoderV2(nn.Module):
    """Encode per-agent features with richer representation."""
    def __init__(self, in_dim=10, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, agents, mask):
        # agents: (B, N, in_dim), mask: (B, N) bool
        return self.mlp(agents)  # (B, N, hidden)


class InteractionModule(nn.Module):
    """Multi-head self-attention for agent-agent interaction."""
    def __init__(self, dim=64, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        # x: (B, N, dim), mask: (B, N) bool → key_padding_mask needs True=ignore
        key_padding_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + attn_out)  # (B, N, dim)


class AttentionPool(nn.Module):
    """Attention-weighted pooling over agents."""
    def __init__(self, dim=64, out_dim=128):
        super().__init__()
        self.attn_weight = nn.Linear(dim, 1)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x, mask):
        # x: (B, N, dim), mask: (B, N)
        scores = self.attn_weight(x).squeeze(-1)  # (B, N)
        scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)  # (B, N)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, dim)
        return self.proj(pooled)  # (B, out_dim)


class CollisionPredictionNetwork(nn.Module):
    """Predict collision probability and TTC from scene."""

    def __init__(self, agent_feat_dim=10, hidden=64, scene_dim=128):
        super().__init__()
        self.encoder = AgentEncoderV2(agent_feat_dim, hidden)
        self.interaction = InteractionModule(hidden, n_heads=4)
        self.pool = AttentionPool(hidden, scene_dim)

        self.collision_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.ttc_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, agents, mask, ego=None):
        """
        Args:
            agents: (B, N, feat_dim) agent features
            mask: (B, N) boolean mask (True=valid)
            ego: (B, ego_dim) optional ego features (unused for now)
        Returns:
            collision_prob: (B, 1) sigmoid probability
            ttc: (B, 1) estimated TTC (relu, seconds)
        """
        x = self.encoder(agents, mask)  # (B, N, hidden)
        x = self.interaction(x, mask)  # (B, N, hidden)
        scene = self.pool(x, mask)  # (B, scene_dim)

        coll_logit = self.collision_head(scene)  # (B, 1)
        coll_prob = torch.sigmoid(coll_logit)

        ttc = F.relu(self.ttc_head(scene))  # (B, 1)

        return coll_prob, ttc

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CollisionPredictionNetwork()
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    B, N = 4, 20
    agents = torch.randn(B, N, 10)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 15:] = False

    coll, ttc = model(agents, mask)
    print(f"collision_prob: {coll.shape}, ttc: {ttc.shape}")
    print(f"collision_prob: {coll.squeeze().detach().numpy()}")
    print(f"ttc: {ttc.squeeze().detach().numpy()}")

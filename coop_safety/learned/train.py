"""Train Risk Assessment Network on rule-generated labels.

Usage:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    python coop_safety/learned/train.py
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coop_safety.learned.risk_network import RiskAssessmentNetwork

LABEL_DIR = "/raid/xuyifan/jiqiuyu/data/risk_labels"
MODEL_DIR = "/raid/xuyifan/jiqiuyu/models"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class RiskDataset(Dataset):
    def __init__(self, label_dir: str, split="train", train_ratio=0.8):
        d = Path(label_dir)
        self.ego = np.load(d / "ego_features.npy")
        self.agents = np.load(d / "agent_features.npy")
        self.agent_counts = np.load(d / "agent_counts.npy")
        self.risk_points = np.load(d / "risk_points.npy")
        self.conflicts = np.load(d / "conflict_labels.npy")
        self.conflict_counts = np.load(d / "conflict_counts.npy")

        n = len(self.ego)
        split_idx = int(n * train_ratio)
        if split == "train":
            self.indices = list(range(split_idx))
        else:
            self.indices = list(range(split_idx, n))

        # Normalize: compute mean/std from training data
        train_ego = self.ego[:split_idx]
        train_agents = self.agents[:split_idx]
        valid_mask = self.agent_counts[:split_idx] > 0

        self.ego_mean = train_ego.mean(axis=0)
        self.ego_std = train_ego.std(axis=0) + 1e-8
        # For agents, compute stats over all valid agents
        all_valid = []
        for i in range(split_idx):
            c = self.agent_counts[i]
            if c > 0:
                all_valid.append(self.agents[i, :c])
        if all_valid:
            all_valid = np.concatenate(all_valid, axis=0)
            self.agent_mean = all_valid.mean(axis=0)
            self.agent_std = all_valid.std(axis=0) + 1e-8
        else:
            self.agent_mean = np.zeros(8)
            self.agent_std = np.ones(8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        ego = (self.ego[idx] - self.ego_mean) / self.ego_std
        agents = (self.agents[idx] - self.agent_mean) / self.agent_std
        n_agents = self.agent_counts[idx]
        mask = np.zeros(agents.shape[0], dtype=bool)
        mask[:n_agents] = True

        # Risk query points (ego-relative)
        rp = self.risk_points[idx]  # (Q, 3): x, y, risk
        ego_x, ego_y = self.ego[idx, 0], self.ego[idx, 1]
        query_xy = rp[:, :2].copy()
        query_xy[:, 0] -= ego_x
        query_xy[:, 1] -= ego_y
        query_risk = rp[:, 2:3]  # (Q, 1)

        # Conflict pairs
        conf = self.conflicts[idx]  # (P, 4): i, j, prob, ttc
        n_conf = self.conflict_counts[idx]
        pair_indices = conf[:, :2]  # (P, 2)
        pair_labels = conf[:, 2:4]  # (P, 2): prob, ttc
        pair_mask = np.zeros(conf.shape[0], dtype=bool)
        pair_mask[:n_conf] = True

        return {
            "ego": torch.FloatTensor(ego),
            "agents": torch.FloatTensor(agents),
            "mask": torch.BoolTensor(mask),
            "query_points": torch.FloatTensor(query_xy),
            "query_risk": torch.FloatTensor(query_risk),
            "pair_indices": torch.FloatTensor(pair_indices),
            "pair_labels": torch.FloatTensor(pair_labels),
            "pair_mask": torch.BoolTensor(pair_mask),
        }


def train():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading dataset...")
    train_ds = RiskDataset(LABEL_DIR, split="train")
    val_ds = RiskDataset(LABEL_DIR, split="val")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    model = RiskAssessmentNetwork(agent_feat_dim=8, embed_dim=32).to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    risk_criterion = nn.MSELoss()
    prob_criterion = nn.BCELoss()
    ttc_criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")

    for epoch in range(50):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            ego = batch["ego"].to(DEVICE)
            agents = batch["agents"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            query_points = batch["query_points"].to(DEVICE)
            query_risk = batch["query_risk"].to(DEVICE)
            pair_indices = batch["pair_indices"].to(DEVICE)
            pair_labels = batch["pair_labels"].to(DEVICE)
            pair_mask = batch["pair_mask"].to(DEVICE)

            out = model(ego, agents, mask, query_points, pair_indices)

            # Risk loss
            loss_risk = risk_criterion(out["risk"], query_risk)

            # Conflict loss (only on valid pairs)
            if pair_mask.any():
                pred_conf = out["conflict"][pair_mask]  # (valid_pairs, 2)
                true_conf = pair_labels[pair_mask]  # (valid_pairs, 2)
                loss_prob = prob_criterion(pred_conf[:, 0], true_conf[:, 0].clamp(0, 1))
                loss_ttc = ttc_criterion(pred_conf[:, 1], true_conf[:, 1])
                loss_conflict = loss_prob + 0.1 * loss_ttc
            else:
                loss_conflict = torch.tensor(0.0, device=DEVICE)

            loss = loss_risk + 0.5 * loss_conflict

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                ego = batch["ego"].to(DEVICE)
                agents = batch["agents"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                query_points = batch["query_points"].to(DEVICE)
                query_risk = batch["query_risk"].to(DEVICE)
                pair_indices = batch["pair_indices"].to(DEVICE)
                pair_labels = batch["pair_labels"].to(DEVICE)
                pair_mask = batch["pair_mask"].to(DEVICE)

                out = model(ego, agents, mask, query_points, pair_indices)
                loss_risk = risk_criterion(out["risk"], query_risk)
                if pair_mask.any():
                    pred_conf = out["conflict"][pair_mask]
                    true_conf = pair_labels[pair_mask]
                    loss_prob = prob_criterion(pred_conf[:, 0], true_conf[:, 0].clamp(0, 1))
                    loss_ttc = ttc_criterion(pred_conf[:, 1], true_conf[:, 1])
                    loss_conflict = loss_prob + 0.1 * loss_ttc
                else:
                    loss_conflict = torch.tensor(0.0)

                val_loss += (loss_risk + 0.5 * loss_conflict).item()
                val_batches += 1

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(val_batches, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
                "ego_mean": train_ds.ego_mean,
                "ego_std": train_ds.ego_std,
                "agent_mean": train_ds.agent_mean,
                "agent_std": train_ds.agent_std,
            }, f"{MODEL_DIR}/risk_net_best.pt")
            marker = " *best*"
        else:
            marker = ""

        if epoch % 5 == 0 or marker:
            print(f"Epoch {epoch:3d}: train={avg_train:.4f} val={avg_val:.4f}{marker}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {MODEL_DIR}/risk_net_best.pt")

    # Save final model too
    torch.save(model.state_dict(), f"{MODEL_DIR}/risk_net_final.pt")


if __name__ == "__main__":
    train()

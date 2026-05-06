"""Train CollisionPredictionNetV2 on DeepAccident.

Key improvements over v1 training:
1. Relative features (ego-centric coordinates + approach speed + dist)
2. Visibility as key feature (invisible agents → higher danger signal)
3. Waypoint risk labels (waypoints near collision → high risk)
4. Multi-task loss: focal_bce(collision) + mse(ttc) + bce(waypoint_risk)
"""
from __future__ import annotations

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coop_safety.learned.collision_network_v2 import CollisionPredictionNetV2
from experiments.deepaccident_loader import DeepAccidentLoader

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/raid/xuyifan/jiqiuyu/models"
MAX_AGENTS = 30
N_WAYPOINTS = 10


class DeepAccidentDatasetV2(Dataset):
    def __init__(self, loader, split='train', train_ratio=0.8):
        self.samples = []

        for si, s in enumerate(loader.scenarios):
            cf = s.get('collision_frame', -1)
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                ftc = cf - fi if cf > 0 and s['is_accident'] else -1

                # Collision label
                is_dangerous = s['is_accident'] and 0 < ftc <= 30
                ttc = ftc * 0.05 if ftc > 0 else 10.0

                ego = frame.perception.ego
                ego_speed = ego.velocity
                ego_yaw = frame.ego_yaw_rate

                # Encode agents with RELATIVE features + visibility
                agents = np.zeros((MAX_AGENTS, 12), dtype=np.float32)
                mask = np.zeros(MAX_AGENTS, dtype=bool)
                for i, a in enumerate(frame.perception.agents[:MAX_AGENTS]):
                    s_a = a.state
                    rel_x = s_a.x  # Already ego-centric in DeepAccident
                    rel_y = s_a.y
                    rel_vx = s_a.vx - ego_speed  # Relative velocity
                    rel_vy = s_a.vy
                    dist = math.sqrt(rel_x**2 + rel_y**2)
                    # Approach speed (negative = approaching)
                    if dist > 0.01:
                        approach = -(rel_x * rel_vx + rel_y * rel_vy) / dist
                    else:
                        approach = 0

                    agents[i] = [
                        rel_x, rel_y, rel_vx, rel_vy,
                        s_a.heading, s_a.length, s_a.width, s_a.velocity,
                        1.0 if a.is_visible else 0.0,  # KEY: visibility
                        0,  # type placeholder
                        approach,  # approach speed
                        dist,  # distance to ego
                    ]
                    mask[i] = True

                # Ego features
                ego_feat = np.array([ego_speed, ego_yaw, 0, 0, 0, 0], dtype=np.float32)

                # Waypoint risk labels: simulate constant-velocity waypoints
                waypoints = np.zeros((N_WAYPOINTS, 2), dtype=np.float32)
                wp_risk = np.zeros((N_WAYPOINTS, 1), dtype=np.float32)
                for t in range(N_WAYPOINTS):
                    dt = (t + 1) * 0.5
                    waypoints[t] = [ego_speed * dt, 0]  # Straight ahead
                    # Check if waypoint is near any agent at time dt
                    for a in frame.perception.agents:
                        ax = a.state.x + a.state.vx * dt
                        ay = a.state.y + a.state.vy * dt
                        wp_dist = math.sqrt((waypoints[t, 0]-ax)**2 + (waypoints[t, 1]-ay)**2)
                        if wp_dist < 3.0:
                            wp_risk[t] = 1.0
                            break

                self.samples.append({
                    'agents': agents, 'mask': mask, 'ego': ego_feat,
                    'waypoints': waypoints, 'wp_risk': wp_risk,
                    'is_dangerous': float(is_dangerous), 'ttc': ttc,
                })

        np.random.seed(42)
        np.random.shuffle(self.samples)
        n = len(self.samples)
        idx = int(n * train_ratio)
        if split == 'train':
            self.samples = self.samples[:idx]
        else:
            self.samples = self.samples[idx:]

        n_pos = sum(1 for s in self.samples if s['is_dangerous'] > 0.5)
        n_wp_risk = sum(s['wp_risk'].sum() for s in self.samples)
        print(f"  {split}: {len(self.samples)} samples, pos={n_pos} ({n_pos/max(len(self.samples),1):.1%}), "
              f"wp_risk_frames={int(n_wp_risk)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {k: torch.FloatTensor(v) if isinstance(v, np.ndarray) else torch.FloatTensor([v])
                for k, v in s.items() if k != 'mask'} | {'mask': torch.BoolTensor(s['mask'])}


def train():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    loader = DeepAccidentLoader(split='all')
    train_ds = DeepAccidentDatasetV2(loader, 'train')
    val_ds = DeepAccidentDatasetV2(loader, 'val')

    # Oversample positives
    pos_idx = [i for i, s in enumerate(train_ds.samples) if s['is_dangerous'] > 0.5]
    neg_idx = [i for i, s in enumerate(train_ds.samples) if s['is_dangerous'] <= 0.5]
    extra = np.random.choice(pos_idx, size=min(len(neg_idx)-len(pos_idx), len(pos_idx)*4), replace=True) if pos_idx else []
    balanced = list(range(len(train_ds))) + extra.tolist()
    sampler = torch.utils.data.SubsetRandomSampler(balanced)
    train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = CollisionPredictionNetV2(agent_feat_dim=12, ego_feat_dim=6, hidden=64, scene_dim=128).to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    def focal_bce(pred, target, gamma=2.0, alpha=0.75):
        bce = F.binary_cross_entropy(pred.clamp(1e-7, 1-1e-7), target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        w = torch.where(target > 0.5, alpha, 1 - alpha)
        return (w * (1 - pt) ** gamma * bce).mean()

    best_auc = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        nb = 0

        for batch in train_loader:
            agents = batch['agents'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            ego = batch['ego'].to(DEVICE)
            wp = batch['waypoints'].to(DEVICE)
            label = batch['is_dangerous'].to(DEVICE)
            ttc_label = batch['ttc'].to(DEVICE)
            wp_risk_label = batch['wp_risk'].to(DEVICE)

            out = model(agents, mask, ego, wp)

            loss_coll = focal_bce(out['collision_prob'], label)
            loss_ttc = F.smooth_l1_loss(out['ttc'], ttc_label) * 0.1
            loss_wp = F.binary_cross_entropy(
                out['waypoint_risk'].clamp(1e-7, 1-1e-7), wp_risk_label) * 0.2

            loss = loss_coll + loss_ttc + loss_wp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            nb += 1

        scheduler.step()

        # Validation
        model.eval()
        preds, labels = [], []
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                agents = batch['agents'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                ego = batch['ego'].to(DEVICE)
                wp = batch['waypoints'].to(DEVICE)
                label = batch['is_dangerous'].to(DEVICE)
                ttc_label = batch['ttc'].to(DEVICE)
                wp_risk_label = batch['wp_risk'].to(DEVICE)

                out = model(agents, mask, ego, wp)
                loss = focal_bce(out['collision_prob'], label) + \
                       F.smooth_l1_loss(out['ttc'], ttc_label)*0.1 + \
                       F.binary_cross_entropy(out['waypoint_risk'].clamp(1e-7,1-1e-7), wp_risk_label)*0.2
                val_loss += loss.item()
                vn += 1
                preds.extend(out['collision_prob'].cpu().numpy().flatten())
                labels.extend(label.cpu().numpy().flatten())

        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        preds = np.array(preds)
        labels = np.array(labels)
        try: auc = roc_auc_score(labels, preds)
        except: auc = 0.5
        pred_bin = (preds > 0.5).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, pred_bin, average='binary', zero_division=0)

        marker = ""
        if auc > best_auc:
            best_auc = auc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'auc': auc},
                       f"{MODEL_DIR}/collision_net_v2_best.pt")
            marker = " *best*"

        if epoch % 5 == 0 or marker:
            print(f"Ep {epoch:3d}: train={train_loss/max(nb,1):.4f} val={val_loss/max(vn,1):.4f} "
                  f"AUC={auc:.4f} P={p:.3f} R={r:.3f} F1={f1:.3f}{marker}")

    print(f"\nBest AUC: {best_auc:.4f}")
    torch.save(model.state_dict(), f"{MODEL_DIR}/collision_net_v2_final.pt")


if __name__ == "__main__":
    train()

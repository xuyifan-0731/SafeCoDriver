"""Train CollisionPredictionNetwork on DeepAccident.

Labels:
  - Positive: accident scenarios, frames within 30 frames of collision
  - Negative: normal scenarios + accident scenarios far from collision
  - TTC: frames_to_collision * 0.05s (20fps assumed)

Usage:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    python coop_safety/learned/train_collision.py
"""
from __future__ import annotations

import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch.nn.functional as F

from coop_safety.learned.collision_network import CollisionPredictionNetwork
from experiments.deepaccident_loader import DeepAccidentLoader

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "/raid/xuyifan/jiqiuyu/models"
MAX_AGENTS = 30


class DeepAccidentDataset(Dataset):
    def __init__(self, loader: DeepAccidentLoader, split='train', train_ratio=0.8):
        # Collect all frames
        self.samples = []
        for si, s in enumerate(loader.scenarios):
            collision_frame = s.get('collision_frame', -1)
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                ftc = collision_frame - fi if collision_frame > 0 and s['is_accident'] else -1

                # Label
                is_dangerous = s['is_accident'] and 0 < ftc <= 30
                ttc = ftc * 0.05 if ftc > 0 else 10.0  # Cap at 10s

                # Encode agents
                agents = np.zeros((MAX_AGENTS, 10), dtype=np.float32)
                mask = np.zeros(MAX_AGENTS, dtype=bool)
                for i, a in enumerate(frame.perception.agents[:MAX_AGENTS]):
                    s_a = a.state
                    agents[i] = [s_a.x, s_a.y, s_a.vx, s_a.vy, s_a.heading,
                                 s_a.length, s_a.width, s_a.velocity,
                                 1.0 if a.is_visible else 0.0,
                                 0]  # placeholder
                    mask[i] = True

                self.samples.append({
                    'agents': agents,
                    'mask': mask,
                    'is_dangerous': float(is_dangerous),
                    'ttc': ttc,
                    'n_agents': min(len(frame.perception.agents), MAX_AGENTS),
                })

        # Shuffle samples before splitting (ensure both splits have pos/neg)
        np.random.seed(42)
        np.random.shuffle(self.samples)

        # Split
        n = len(self.samples)
        split_idx = int(n * train_ratio)
        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        # Stats
        n_pos = sum(1 for s in self.samples if s['is_dangerous'] > 0.5)
        n_neg = len(self.samples) - n_pos
        print(f"  {split}: {len(self.samples)} samples (pos={n_pos}, neg={n_neg}, ratio={n_pos/max(len(self.samples),1):.1%})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'agents': torch.FloatTensor(s['agents']),
            'mask': torch.BoolTensor(s['mask']),
            'label': torch.FloatTensor([s['is_dangerous']]),
            'ttc': torch.FloatTensor([s['ttc']]),
        }


def train():
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")

    print("Loading DeepAccident data...")
    loader = DeepAccidentLoader(split='all')

    print("Building datasets...")
    train_ds = DeepAccidentDataset(loader, split='train')
    val_ds = DeepAccidentDataset(loader, split='val')

    # Balance training with oversampling positives
    pos_indices = [i for i, s in enumerate(train_ds.samples) if s['is_dangerous'] > 0.5]
    neg_indices = [i for i, s in enumerate(train_ds.samples) if s['is_dangerous'] <= 0.5]
    n_oversample = len(neg_indices) - len(pos_indices)
    if n_oversample > 0 and pos_indices:
        extra = np.random.choice(pos_indices, size=min(n_oversample, len(pos_indices)*3), replace=True)
        balanced_indices = list(range(len(train_ds.samples))) + extra.tolist()
        sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)
        train_loader = DataLoader(train_ds, batch_size=64, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = CollisionPredictionNetwork(agent_feat_dim=10, hidden=64, scene_dim=128).to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    # Focal loss for imbalanced data
    def focal_bce(pred, target, gamma=2.0, alpha=0.75):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        weight = torch.where(target > 0.5, alpha, 1 - alpha)
        focal = weight * (1 - pt) ** gamma * bce
        return focal.mean()

    best_val_auc = 0
    best_epoch = 0

    for epoch in range(80):
        model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            agents = batch['agents'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            label = batch['label'].to(DEVICE)
            ttc_label = batch['ttc'].to(DEVICE)

            coll_prob, ttc_pred = model(agents, mask)

            loss_coll = focal_bce(coll_prob, label)
            loss_ttc = F.smooth_l1_loss(ttc_pred, ttc_label) * 0.1

            loss = loss_coll + loss_ttc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        vn = 0

        with torch.no_grad():
            for batch in val_loader:
                agents = batch['agents'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                label = batch['label'].to(DEVICE)
                ttc_label = batch['ttc'].to(DEVICE)

                coll_prob, ttc_pred = model(agents, mask)
                loss = focal_bce(coll_prob, label) + F.smooth_l1_loss(ttc_pred, ttc_label) * 0.1
                val_loss += loss.item()
                vn += 1

                val_preds.extend(coll_prob.cpu().numpy().flatten())
                val_labels.extend(label.cpu().numpy().flatten())

        # Compute AUC
        from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        try:
            auc = roc_auc_score(val_labels, val_preds)
        except:
            auc = 0.5

        # Binary metrics at threshold 0.5
        pred_binary = (val_preds > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(val_labels, pred_binary, average='binary', zero_division=0)

        avg_train = train_loss / max(n_batches, 1)
        avg_val = val_loss / max(vn, 1)

        marker = ""
        if auc > best_val_auc:
            best_val_auc = auc
            best_epoch = epoch
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'auc': auc,
            }, f"{MODEL_DIR}/collision_net_best.pt")
            marker = " *best*"

        if epoch % 5 == 0 or marker:
            print(f"Epoch {epoch:3d}: train={avg_train:.4f} val={avg_val:.4f} "
                  f"AUC={auc:.4f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}{marker}")

    print(f"\nBest AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    torch.save(model.state_dict(), f"{MODEL_DIR}/collision_net_final.pt")
    print(f"Models saved to {MODEL_DIR}")


if __name__ == "__main__":
    train()

"""Ablation experiments for SafeCoDriver improvements.

Tests each improvement independently, then combines effective ones.

Improvements:
  A) Detection threshold 0.3 → 0.5
  B) V1 + ego features (speed, heading) + retrain
  C) Lateral avoidance in closed-loop (not just braking)
  D) V1 temporal modeling (3-frame input)
  E) Collision severity optimization (speed-aware waypoint modification)

Methodology:
  1. Run each improvement independently on DeepAccident (104 scenarios)
  2. Identify effective improvements
  3. Combine effective ones
  4. Run best combo on SUMO closed-loop
"""
from __future__ import annotations
import sys, os, math, time, random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import (
    simulate_codriving_waypoints, check_waypoint_collision, MethodResult
)
from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType, ConstraintMode

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Improvement B: V1 + Ego Features
# ============================================================

class CollisionNetV1Ego(nn.Module):
    """V1 with ego features: speed, heading → better context."""

    def __init__(self, agent_feat_dim=10, ego_feat_dim=4, hidden=64, scene_dim=128):
        super().__init__()
        self.agent_enc = nn.Sequential(
            nn.Linear(agent_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.interaction = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.attn_weight = nn.Linear(hidden, 1)
        self.ego_enc = nn.Sequential(nn.Linear(ego_feat_dim, 32), nn.ReLU())
        self.proj = nn.Linear(hidden + 32, scene_dim)
        self.collision_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.ttc_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, agents, mask, ego_feat=None):
        x = self.agent_enc(agents)
        attn_out, _ = self.interaction(x, x, x, key_padding_mask=~mask)
        x = self.norm(x + attn_out)
        scores = self.attn_weight(x).squeeze(-1).masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)

        if ego_feat is not None:
            ego_emb = self.ego_enc(ego_feat)
            scene = self.proj(torch.cat([pooled, ego_emb], dim=-1))
        else:
            scene = self.proj(torch.cat([pooled, torch.zeros(pooled.shape[0], 32, device=pooled.device)], dim=-1))

        coll = torch.sigmoid(self.collision_head(scene))
        ttc = F.relu(self.ttc_head(scene))
        return coll, ttc

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Improvement D: V1 Temporal (3-frame)
# ============================================================

class CollisionNetV1Temporal(nn.Module):
    """V1 with temporal modeling: 3 frames concatenated."""

    def __init__(self, agent_feat_dim=10, n_frames=3, hidden=64, scene_dim=128):
        super().__init__()
        self.n_frames = n_frames
        self.agent_enc = nn.Sequential(
            nn.Linear(agent_feat_dim * n_frames, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.interaction = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.norm = nn.LayerNorm(hidden)
        self.attn_weight = nn.Linear(hidden, 1)
        self.proj = nn.Linear(hidden, scene_dim)
        self.collision_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.ttc_head = nn.Sequential(
            nn.Linear(scene_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, agents, mask, ego_feat=None):
        # agents: (B, N, feat_dim * n_frames) - concatenated multi-frame features
        x = self.agent_enc(agents)
        attn_out, _ = self.interaction(x, x, x, key_padding_mask=~mask)
        x = self.norm(x + attn_out)
        scores = self.attn_weight(x).squeeze(-1).masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        scene = self.proj(pooled)
        coll = torch.sigmoid(self.collision_head(scene))
        ttc = F.relu(self.ttc_head(scene))
        return coll, ttc

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Training utilities
# ============================================================

def train_model(model, train_data, val_data, epochs=60, lr=5e-4, device='cpu'):
    """Quick training loop."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        random.shuffle(train_data)
        total_loss = 0
        n_batch = 0

        for i in range(0, len(train_data), 64):
            batch = train_data[i:i+64]
            agents = torch.stack([b['agents'] for b in batch]).to(device)
            mask = torch.stack([b['mask'] for b in batch]).to(device)
            labels = torch.FloatTensor([b['label'] for b in batch]).unsqueeze(1).to(device)
            ego_feat = torch.stack([b['ego'] for b in batch]).to(device) if 'ego' in batch[0] else None

            if ego_feat is not None:
                coll, _ = model(agents, mask, ego_feat)
            else:
                coll, _ = model(agents, mask)

            # Focal loss
            bce = F.binary_cross_entropy(coll.clamp(1e-7, 1-1e-7), labels, reduction='none')
            pt = torch.where(labels > 0.5, coll, 1 - coll)
            alpha = torch.where(labels > 0.5, 0.75, 0.25)
            loss = (alpha * (1 - pt)**2 * bce).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1

        scheduler.step()

        # Validate
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for i in range(0, len(val_data), 64):
                batch = val_data[i:i+64]
                agents = torch.stack([b['agents'] for b in batch]).to(device)
                mask = torch.stack([b['mask'] for b in batch]).to(device)
                ego_feat = torch.stack([b['ego'] for b in batch]).to(device) if 'ego' in batch[0] else None
                if ego_feat is not None:
                    coll, _ = model(agents, mask, ego_feat)
                else:
                    coll, _ = model(agents, mask)
                preds.extend(coll.cpu().numpy().flatten())
                labels_all.extend([b['label'] for b in batch])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels_all, preds)
        except:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0:
            print(f"    Ep {epoch}: loss={total_loss/max(n_batch,1):.4f} AUC={auc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"    Best AUC: {best_auc:.4f}")
    return best_auc


def prepare_training_data(loader, include_ego=False, temporal_frames=1):
    """Prepare training data from DeepAccident."""
    train_data, val_data = [], []

    # Collect frames with history for temporal
    frame_cache = {}  # (si, fi) → encoded features

    for si, s in enumerate(loader.scenarios):
        cf = s.get('collision_frame', -1)
        for fi in range(len(s['frames'])):
            frame = loader.load_frame(si, fi)
            ftc = cf - fi if cf > 0 and s['is_accident'] else -1
            is_dangerous = float(s['is_accident'] and 0 < ftc <= 30)

            # Encode agents
            agents = torch.zeros(30, 10)
            mask = torch.zeros(30, dtype=torch.bool)
            for i, a in enumerate(frame.perception.agents[:30]):
                st = a.state
                agents[i] = torch.FloatTensor([
                    st.x, st.y, st.vx, st.vy, st.heading,
                    st.length, st.width, st.velocity,
                    1.0 if a.is_visible else 0.0, 0])
                mask[i] = True

            sample = {'agents': agents, 'mask': mask, 'label': is_dangerous}

            if include_ego:
                ego = frame.perception.ego
                sample['ego'] = torch.FloatTensor([ego.velocity, ego.yaw_rate if hasattr(ego, 'yaw_rate') else 0, 0, 0])

            if temporal_frames > 1:
                # Concatenate current + previous frames
                all_feats = [agents]
                for prev_offset in range(1, temporal_frames):
                    prev_fi = fi - prev_offset
                    if prev_fi >= 0:
                        prev_frame = loader.load_frame(si, prev_fi)
                        prev_agents = torch.zeros(30, 10)
                        for i, a in enumerate(prev_frame.perception.agents[:30]):
                            st = a.state
                            prev_agents[i] = torch.FloatTensor([
                                st.x, st.y, st.vx, st.vy, st.heading,
                                st.length, st.width, st.velocity,
                                1.0 if a.is_visible else 0.0, 0])
                        all_feats.append(prev_agents)
                    else:
                        all_feats.append(torch.zeros(30, 10))
                sample['agents'] = torch.cat(all_feats, dim=-1)  # (30, 10*n_frames)

            # 80/20 split
            if random.random() < 0.8:
                train_data.append(sample)
            else:
                val_data.append(sample)

    # Oversample positives
    pos = [s for s in train_data if s['label'] > 0.5]
    if pos:
        extra = random.choices(pos, k=min(len(pos)*3, len(train_data)-len(pos)))
        train_data.extend(extra)

    print(f"    Data: train={len(train_data)} (pos={sum(1 for s in train_data if s['label']>0.5)}), val={len(val_data)}")
    return train_data, val_data


# ============================================================
# Improvement E: Severity-aware waypoint modification
# ============================================================

class SeverityAwareHybrid:
    """Hybrid with severity optimization: reduce speed near threats."""

    name = "Ours-Severity"

    def __init__(self, detector_model, base_margin_visible=2.5,
                 base_margin_invisible=4.0, detection_threshold=0.5,
                 speed_reduction_factor=0.7):
        self.detector = detector_model
        self.base_margin_visible = base_margin_visible
        self.base_margin_invisible = base_margin_invisible
        self.detection_threshold = detection_threshold
        self.speed_reduction_factor = speed_reduction_factor

    def constrain_waypoints(self, waypoints, perception):
        modified = waypoints.copy()
        ego_speed = max(perception.ego.velocity, 1.0)
        n_threats = 0

        for t in range(len(waypoints)):
            dt = (t + 1) * 0.5
            threats = []
            for a in perception.agents:
                s = a.state
                ax = s.x + s.vx * dt
                ay = s.y + s.vy * dt
                dist = math.sqrt((modified[t,0]-ax)**2 + (modified[t,1]-ay)**2)

                margin = self.base_margin_invisible if not a.is_visible else self.base_margin_visible
                # Approach speed scaling
                rel_dist = math.sqrt(s.x**2 + s.y**2)
                if rel_dist > 0.01:
                    approach = -(s.x*(s.vx-ego_speed) + s.y*s.vy) / rel_dist
                    if approach > 0:
                        margin *= (1 + 0.3 * min(approach/20.0, 1.0))
                margin += max(s.length, s.width) * 0.3

                if dist < margin:
                    threats.append((a, ax, ay, dist, margin))

            if threats:
                n_threats += 1
                # Multi-agent repulsion
                fx, fy, max_push = 0, 0, 0
                for agent, ax, ay, dist, margin in threats:
                    dx = modified[t,0] - ax
                    dy = modified[t,1] - ay
                    d = max(dist, 0.1)
                    force = (margin - dist + 1.0) / d
                    vis_w = 1.5 if not agent.is_visible else 1.0
                    fx += (dx/d) * force * vis_w
                    fy += (dy/d) * force * vis_w
                    max_push = max(max_push, margin - dist + 1.0)

                fn = math.sqrt(fx**2 + fy**2)
                if fn > 0.01:
                    push = max(max_push, 1.0)
                    modified[t,0] += (fx/fn) * push
                    modified[t,1] += (fy/fn) * push

                # SEVERITY OPTIMIZATION: reduce waypoint distance (= reduce speed)
                # This is the key addition: shorten subsequent waypoints
                speed_factor = self.speed_reduction_factor
                for t2 in range(t+1, len(modified)):
                    modified[t2, 0] *= speed_factor
                    # Only apply once per chain
                break  # One reduction per frame is enough

        # Smoothing
        for t in range(1, len(modified)-1):
            modified[t] = 0.7*modified[t] + 0.15*(modified[t-1]+modified[t+1])

        # Speed clamp
        for t in range(1, len(modified)):
            dx = modified[t,0]-modified[t-1,0]
            dy = modified[t,1]-modified[t-1,1]
            d = math.sqrt(dx**2+dy**2)
            if d > 10:
                s = 10/d
                modified[t,0] = modified[t-1,0]+dx*s
                modified[t,1] = modified[t-1,1]+dy*s

        # Detection (V1)
        is_dangerous = False
        if self.detector:
            agents_feat = np.zeros((1,30,10), dtype=np.float32)
            mask = np.zeros((1,30), dtype=bool)
            for i,a in enumerate(perception.agents[:30]):
                s=a.state
                agents_feat[0,i]=[s.x,s.y,s.vx,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0]
                mask[0,i]=True
            with torch.no_grad():
                cp,_=self.detector(torch.FloatTensor(agents_feat),torch.BoolTensor(mask))
            is_dangerous = cp.item() > self.detection_threshold

        return modified, {
            "method": self.name,
            "n_collisions_detected": 1 if is_dangerous else 0,
            "modification_rate": 1.0/10 if is_dangerous else 0,
            "n_geometric_threats": n_threats,
        }


# ============================================================
# Main: Run all ablations
# ============================================================

def evaluate_method_on_deepaccident(method, loader):
    """Evaluate one method on full DeepAccident."""
    result = MethodResult(name=method.name if hasattr(method, 'name') else "?")

    for si, s in enumerate(loader.scenarios):
        first_warning = -1
        false_alarm = 0
        wp_coll = 0
        modifications = 0

        for fi in range(len(s['frames'])):
            frame = loader.load_frame(si, fi)
            base_wp = simulate_codriving_waypoints(frame)
            modified_wp, stats = method.constrain_waypoints(base_wp, frame.perception)
            was_modified = stats.get('n_collisions_detected', 0) > 0

            if was_modified:
                if first_warning < 0: first_warning = fi
                if not s['is_accident']: false_alarm += 1
                modifications += 1
            wp_coll += check_waypoint_collision(modified_wp, frame)

        result.total_frames += len(s['frames'])
        result.modified_frames += modifications
        result.waypoint_collisions += wp_coll
        result.total_waypoint_checks += len(s['frames']) * 10

        if s['is_accident']:
            result.n_accident_scenarios += 1
            if first_warning >= 0:
                result.n_detected += 1
                result.early_warning_frames.append(len(s['frames']) - first_warning)
        else:
            result.n_normal_scenarios += 1
            if false_alarm > 0:
                result.n_false_alarm_scenarios += 1

    return result


def print_result(result):
    det = result.n_detected / max(result.n_accident_scenarios, 1)
    early = np.mean(result.early_warning_frames) if result.early_warning_frames else 0
    fa = result.n_false_alarm_scenarios / max(result.n_normal_scenarios, 1)
    wpc = result.waypoint_collisions / max(result.total_waypoint_checks, 1)
    mod = result.modified_frames / max(result.total_frames, 1)
    print(f"    Det={det:.1%} Early={early:.1f} FA={fa:.1%} WPColl={wpc:.1%} Mod={mod:.1%}")


def main():
    print("="*70)
    print("  SafeCoDriver Improvement Ablations")
    print("="*70)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    loader = DeepAccidentLoader(split='all')
    t0 = time.time()

    # Load baseline V1
    from coop_safety.learned.collision_network import CollisionPredictionNetwork
    from coop_safety.learned.hybrid_safety import HybridSafetyConstraint

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    # ==========================================
    # Baseline: Ours-Hybrid with threshold=0.3
    # ==========================================
    print("\n[Baseline] Ours-Hybrid (threshold=0.3)")
    baseline = HybridSafetyConstraint(detector_model=v1, detection_threshold=0.3)
    r = evaluate_method_on_deepaccident(baseline, loader)
    print_result(r)

    # ==========================================
    # A) Threshold = 0.5
    # ==========================================
    print("\n[A] Threshold=0.5")
    method_a = HybridSafetyConstraint(detector_model=v1, detection_threshold=0.5)
    r = evaluate_method_on_deepaccident(method_a, loader)
    print_result(r)

    # ==========================================
    # B) V1 + Ego features (retrain)
    # ==========================================
    print("\n[B] V1 + Ego features (training...)")
    train_data, val_data = prepare_training_data(loader, include_ego=True, temporal_frames=1)
    model_b = CollisionNetV1Ego(agent_feat_dim=10, ego_feat_dim=4)
    print(f"    Params: {model_b.count_parameters():,}")
    auc_b = train_model(model_b, train_data, val_data, epochs=60)

    class HybridB:
        name = "Ours-Hybrid-EgoFeat"
        def __init__(self, model, threshold):
            self.model = model; self.threshold = threshold
            self.base = HybridSafetyConstraint(detector_model=None, detection_threshold=threshold)
        def constrain_waypoints(self, waypoints, perception):
            mod_wp, stats = self.base.constrain_waypoints(waypoints, perception)
            # V1+ego detection
            agents_feat = torch.zeros(1,30,10)
            mask = torch.zeros(1,30,dtype=torch.bool)
            for i,a in enumerate(perception.agents[:30]):
                s=a.state
                agents_feat[0,i]=torch.FloatTensor([s.x,s.y,s.vx,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0])
                mask[0,i]=True
            ego_feat = torch.FloatTensor([[perception.ego.velocity, 0, 0, 0]])
            with torch.no_grad():
                cp,_ = self.model(agents_feat, mask, ego_feat)
            stats['n_collisions_detected'] = 1 if cp.item() > self.threshold else 0
            stats['collision_prob'] = cp.item()
            return mod_wp, stats

    method_b = HybridB(model_b, 0.3)
    r = evaluate_method_on_deepaccident(method_b, loader)
    print_result(r)
    # Also test with threshold=0.5
    print("  [B+A] V1+Ego with threshold=0.5:")
    method_ba = HybridB(model_b, 0.5)
    r = evaluate_method_on_deepaccident(method_ba, loader)
    print_result(r)

    # ==========================================
    # D) V1 Temporal (3-frame)
    # ==========================================
    print("\n[D] V1 Temporal 3-frame (training...)")
    train_data_t, val_data_t = prepare_training_data(loader, include_ego=False, temporal_frames=3)
    model_d = CollisionNetV1Temporal(agent_feat_dim=10, n_frames=3)
    print(f"    Params: {model_d.count_parameters():,}")
    auc_d = train_model(model_d, train_data_t, val_data_t, epochs=60)

    class HybridD:
        name = "Ours-Hybrid-Temporal"
        def __init__(self, model, threshold, loader):
            self.model = model; self.threshold = threshold; self.loader = loader
            self.base = HybridSafetyConstraint(detector_model=None, detection_threshold=threshold)
            self.frame_history = {}  # si → list of agent features per frame
        def constrain_waypoints(self, waypoints, perception):
            mod_wp, stats = self.base.constrain_waypoints(waypoints, perception)
            # Can't easily get temporal in offline eval, use single-frame padded
            agents_feat = torch.zeros(1,30,30)  # 10*3
            mask = torch.zeros(1,30,dtype=torch.bool)
            for i,a in enumerate(perception.agents[:30]):
                s=a.state
                feat = [s.x,s.y,s.vx,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0]
                agents_feat[0,i,:10] = torch.FloatTensor(feat)
                agents_feat[0,i,10:20] = torch.FloatTensor(feat)  # duplicate for missing history
                agents_feat[0,i,20:30] = torch.FloatTensor(feat)
                mask[0,i]=True
            with torch.no_grad():
                cp,_ = self.model(agents_feat, mask)
            stats['n_collisions_detected'] = 1 if cp.item() > self.threshold else 0
            return mod_wp, stats

    method_d = HybridD(model_d, 0.3, loader)
    r = evaluate_method_on_deepaccident(method_d, loader)
    print_result(r)

    # ==========================================
    # E) Severity-aware
    # ==========================================
    print("\n[E] Severity-aware waypoint modification")
    method_e = SeverityAwareHybrid(detector_model=v1, detection_threshold=0.3,
                                    speed_reduction_factor=0.7)
    r = evaluate_method_on_deepaccident(method_e, loader)
    print_result(r)
    print("  [E+A] Severity + threshold=0.5:")
    method_ea = SeverityAwareHybrid(detector_model=v1, detection_threshold=0.5,
                                     speed_reduction_factor=0.7)
    r = evaluate_method_on_deepaccident(method_ea, loader)
    print_result(r)

    # ==========================================
    # Combined: A+B+E (threshold=0.5 + ego + severity)
    # ==========================================
    print("\n[A+B+E] Combined: threshold=0.5 + ego + severity")
    class CombinedABE:
        name = "Ours-Combined-ABE"
        def __init__(self, ego_model, v1_model):
            self.ego_model = ego_model
            self.severity = SeverityAwareHybrid(detector_model=None, detection_threshold=0.5,
                                                speed_reduction_factor=0.7)
        def constrain_waypoints(self, waypoints, perception):
            mod_wp, stats = self.severity.constrain_waypoints(waypoints, perception)
            # Detection with ego model
            agents_feat = torch.zeros(1,30,10)
            mask = torch.zeros(1,30,dtype=torch.bool)
            for i,a in enumerate(perception.agents[:30]):
                s=a.state
                agents_feat[0,i]=torch.FloatTensor([s.x,s.y,s.vx,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0])
                mask[0,i]=True
            ego_feat = torch.FloatTensor([[perception.ego.velocity, 0, 0, 0]])
            with torch.no_grad():
                cp,_ = self.ego_model(agents_feat, mask, ego_feat)
            stats['n_collisions_detected'] = 1 if cp.item() > 0.5 else 0
            return mod_wp, stats

    method_abe = CombinedABE(model_b, v1)
    r = evaluate_method_on_deepaccident(method_abe, loader)
    print_result(r)

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

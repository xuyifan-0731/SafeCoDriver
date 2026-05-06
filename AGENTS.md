# AGENTS.md — Development Guide

> For AI agents or new developers to quickly understand and continue work.

## Project Info

- **Repo**: SafeCoDriver
- **Conda env**: `coop-safety` (Python 3.10, PyTorch 2.5.1+cu121)
- **Dataset**: DeepAccident val (104 scenarios, 8193 frames, ~93GB)
- **Models**: V1 (46K, AUC=0.985), V2 (66K, AUC=0.978)

## Quick Commands

```bash
conda activate coop-safety

# Train models
python coop_safety/learned/train_collision.py      # V1 (~2 min)
python coop_safety/learned/train_collision_v2.py    # V2 (~3 min)

# Run main evaluation
python experiments/run_deepaccident_unified.py      # ~90 min

# Run supplementary experiments
python experiments/run_supplementary.py             # ~11 min
```

## Architecture

```
SafeCoDriver = V1 Detector + Geometric Waypoint Corrector

V1 Detector (46K params):
  AgentEncoder(10→64) → SelfAttention(64,4h) → AttentionPool(64→128) → CollisionHead(128→1)
  Output: P(collision) — used for detection signal only

Geometric Corrector:
  For each waypoint:
    1. Find threats: agents within visibility-aware margin
    2. Multi-agent repulsion: sum forces from ALL threats
    3. Push waypoint away from combined threat direction
  Always runs regardless of V1 output
```

## Key Design Decision

**Detection and correction are DECOUPLED:**
- Waypoints are ALWAYS modified by geometric checking → lowest WPColl%
- Detection flag is set by V1 network only → lowest FalseAlm
- This avoids the trade-off that plagues all other methods

## Files to Read First

| File | Purpose |
|------|---------|
| `coop_safety/learned/hybrid_safety.py` | Final method implementation |
| `coop_safety/learned/collision_network.py` | V1 network architecture |
| `experiments/run_deepaccident_unified.py` | Main evaluation script |
| `experiments/deepaccident_loader.py` | Dataset loading |
| `docs/260429方法试验说明.md` | Complete documentation (Chinese) |

## Current Results (Best)

| Metric | SafeCoDriver | Best Baseline |
|--------|-------------|---------------|
| Detection Rate | 100% | 100% (UniE2EV2X) |
| WP Collision Rate | **0.2%** | 0.4% (MAP) |
| False Alarm Rate | **26.9%** | 94.2% (MAP) |
| Latency | **1.8ms** | 4.9ms (RiskMM) |

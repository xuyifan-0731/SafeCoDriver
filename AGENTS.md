# AGENTS.md — Development Guide

> For AI agents or new developers to quickly understand and continue work.

## Project Info

- **Repo**: SafeCoDriver
- **Conda env**: `coop-safety` (Python 3.10, PyTorch 2.5.1+cu121)
- **Dataset**: DeepAccident val (104 scenarios, 8193 frames, ~93GB)
- **Models**: V1 (46K, current checkpoint AUC=0.814 on scenario-split val). Older docs may mention higher AUC; use checkpoint metadata and current eval outputs as source of truth.

## Quick Commands

```bash
conda activate coop-safety

# Train models
python coop_safety/learned/train_collision.py      # V1 (~2 min)
python coop_safety/learned/train_collision_v2.py    # V2 (~3 min)

# Run current DeepAccident validation metrics
python experiments/run_deepaccident_unified_metrics.py

# Run current SUMO key comparisons
python experiments/run_modified_sumo_comparison.py --scenario-set base --key-methods --out-dir results/modified_sumo_v7_base_key
python experiments/run_modified_sumo_comparison.py --scenario-set stress --key-methods --out-dir results/modified_sumo_v7_stress_key

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
| `experiments/run_deepaccident_unified_metrics.py` | Current DeepAccident validation metrics |
| `experiments/run_modified_sumo_comparison.py` | Current SUMO base/stress metrics |
| `experiments/deepaccident_loader.py` | Dataset loading |
| `docs/260519最优结果代码分析说明.md` | Current code-based method analysis |

## Current Results (Best)

| Setting | Current Best / Key Result |
|---------|---------------------------|
| DeepAccident WPC% | Hybrid family: **0.3%** |
| DeepAccident FA(f) | MAP: **32.6%**; Hybrid+AND: 40.2%; Hybrid-thr0.30: 57.3% |
| SUMO base | Hybrid+AND+TTC / MinHarm / RearEscape: **0% CollRate**, 0 second collisions |
| SUMO stress | Hybrid+AND+TTC+RearEscape: **25% CollRate**, 0 second collisions |

See `results/deepaccident_unified_metrics/`, `results/modified_sumo_v7_base_key/`,
and `results/modified_sumo_v7_stress_key/` for current CSV outputs.

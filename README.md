# SafeCoDriver

**A Pluggable Safety Constraint Framework Based on Cooperative Perception for Autonomous Driving**

基于协同感知的可插拔安全约束框架

---

## Overview

SafeCoDriver is an independent, pluggable safety constraint module that can be stacked on top of any existing autonomous driving planning algorithm. It takes cooperative perception (V2X) results and planner-predicted waypoints as input, and outputs safety-corrected waypoints.

**Key Design: Detection-Correction Decoupling**
- **Collision Detection**: Lightweight neural network (V1, 46K params) → 100% detection rate, 26.9% false alarm
- **Waypoint Correction**: Visibility-aware geometric method → 0.2% waypoint collision rate (lowest among all methods)

## Key Results (DeepAccident, 104 scenarios)

| Method | DetRate↑ | FalseAlm↓ | WPColl%↓ | ms/frame↓ |
|--------|---------|----------|---------|-----------|
| UniE2EV2X [Li24] | 100% | 100% | 0.7% | 10.2 |
| MAP [Yin25] | 96.2% | 94.2% | 0.4% | 6.6 |
| RiskMM [Lei25] | 100% | 100% | 2.7% | 4.9 |
| **SafeCoDriver (Ours)** | **100%** | **26.9%** | **0.2%** | **1.8** |

## Innovations

1. **Visibility-Aware Safety Buffer** — Invisible agents (V2X-only) get 4.0m margin vs 2.5m for visible ones
2. **Approach-Speed Adaptive Collision Zone** — Safety margin scales with agent approach speed
3. **Multi-Agent Repulsive Field** — Sums repulsive forces from ALL threats (prevents pushing into another agent)

## Installation

```bash
conda create -n coop-safety python=3.10 -y
conda activate coop-safety
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy shapely scikit-learn
```

## Data Preparation

Download [DeepAccident](https://github.com/tianqi-wang1996/DeepAccident) val split (~93GB):

```bash
mkdir -p data/DeepAccident
# Download val_part1.zip and val_part2.zip from the official link
# Unzip into data/DeepAccident/
```

Expected structure:
```
data/DeepAccident/
├── type1_subtype1_accident/
│   ├── ego_vehicle/label/
│   ├── other_vehicle/label/
│   └── meta/
├── type1_subtype1_normal/
├── type1_subtype2_accident/
└── type1_subtype2_normal/
```

## Training

```bash
# Train V1 collision prediction network (~2 min on 1x GPU)
python coop_safety/learned/train_collision.py
# Output: models/collision_net_best.pt (AUC=0.985)

# Train V2 improved network (~3 min on 1x GPU)
python coop_safety/learned/train_collision_v2.py
# Output: models/collision_net_v2_best.pt (AUC=0.978)
```

## Evaluation

```bash
# Main experiment: 10 methods x 104 scenarios (~90 min, CPU)
python experiments/run_deepaccident_unified.py

# Supplementary experiments: ablation + efficiency (~11 min)
python experiments/run_supplementary.py
```

## Quick Start

```python
import torch
import numpy as np
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.interface import PerceptionResult, VehicleState, Agent

# Load detector
v1 = CollisionPredictionNetwork()
v1.load_state_dict(torch.load("models/collision_net_best.pt",
                               map_location='cpu', weights_only=False)['model'])
v1.eval()

# Create safety constraint
safety = HybridSafetyConstraint(detector_model=v1)

# Input: perception + waypoints from upstream planner
ego = VehicleState(id='ego', x=0, y=0, heading=0, velocity=10, vx=10, vy=0)
agents = [Agent(state=VehicleState(id='car1', x=15, y=2, heading=3.14,
                                    velocity=8, vx=-8, vy=0), is_visible=False)]
perception = PerceptionResult(timestamp=0, ego=ego, agents=agents)
waypoints = np.array([[5*(t+1), 0.0] for t in range(10)])

# Apply safety constraint
safe_waypoints, stats = safety.constrain_waypoints(waypoints, perception)
print(f"Collision prob: {stats['collision_prob']:.2f}")
print(f"Geometric threats: {stats['n_geometric_threats']}")
```

## Project Structure

```
SafeCoDriver/
├── coop_safety/                    # Core algorithm package
│   ├── interface.py                # Main API: SafetyConstraintModule, data structures
│   ├── learned/
│   │   ├── collision_network.py    # V1: CollisionPredictionNetwork (46K params)
│   │   ├── collision_network_v2.py # V2: CollisionPredictionNetV2 (66K params)
│   │   ├── hybrid_safety.py       # HybridSafetyConstraint (final method)
│   │   ├── train_collision.py     # V1 training script
│   │   └── train_collision_v2.py  # V2 training script
│   ├── perception/                 # Perception: dynamics, prediction, blind spots
│   ├── risk/                       # Three-layer risk: RiskMap, RiskGraph, RiskEvents
│   ├── constraint/                 # Constraints: feasible region, hierarchical, min-harm
│   └── utils/                      # Metrics, visualization
├── experiments/                    # Evaluation scripts
│   ├── deepaccident_loader.py      # DeepAccident dataset loader
│   ├── methods.py                  # Baselines: RSS, CBF
│   ├── methods_modern.py           # Baseline: APF
│   ├── methods_new_baselines.py    # Baselines: UniE2EV2X, MAP, RiskMM
│   ├── run_deepaccident_unified.py # Main evaluation (10 methods)
│   └── run_supplementary.py        # Ablation experiments
├── paper/                          # Paper draft
│   └── SafeCoDriver_draft.md
├── docs/                           # Documentation (Chinese)
├── data/                           # Datasets (not in repo, download separately)
└── models/                         # Trained weights (not in repo, retrain)
```

## Documentation

Detailed Chinese documentation is available in `docs/`:
- `260429方法试验说明.md` — Complete method + experiment documentation
- `README_SafeCoDriver.md` — Algorithm description + reproduction guide

## License

MIT

"""Generate training labels for Risk Assessment Network using rule-based system.

For each frame in DAIR-V2X:
  1. Run rule-based RiskMap → per-cell risk scores (spatial risk labels)
  2. Run rule-based RiskGraph → per-pair conflict scores (interaction labels)
  3. Save as training dataset

Output: data/risk_labels/
  - scene_features.npy: (N_frames, N_max_agents, feature_dim)
  - ego_features.npy: (N_frames, feature_dim)
  - risk_labels.npy: (N_frames, N_query_points, 1)  per-point risk
  - conflict_labels.npy: (N_frames, N_max_pairs, 3)  (prob, ttc, type)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.dairv2x_loader_v2 import DAIRv2xLoaderV2
from coop_safety.risk.risk_map import RiskMapBuilder
from coop_safety.risk.risk_graph import RiskGraphBuilder
from coop_safety.interface import RiskLevel

DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
OUTPUT_DIR = "/raid/xuyifan/jiqiuyu/data/risk_labels"
MAX_AGENTS = 40
FEATURE_DIM = 8  # x, y, vx, vy, heading, length, width, type_id
N_QUERY_POINTS = 100  # spatial query points per frame for risk regression
MAX_PAIRS = 50


def encode_type(vehicle_type: str) -> float:
    return {"car": 0, "truck": 1, "bus": 2, "pedestrian": 3, "bicycle": 4, "motorcycle": 5}.get(vehicle_type, 0)


def main():
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    loader = DAIRv2xLoaderV2(DATA_DIR)
    risk_map_builder = RiskMapBuilder()
    risk_graph_builder = RiskGraphBuilder()

    n_frames = len(loader)
    print(f"Generating labels for {n_frames} frames...")

    all_ego = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
    all_agents = np.zeros((n_frames, MAX_AGENTS, FEATURE_DIM), dtype=np.float32)
    all_agent_counts = np.zeros(n_frames, dtype=np.int32)
    all_risk_points = np.zeros((n_frames, N_QUERY_POINTS, 3), dtype=np.float32)  # x, y, risk
    all_conflict = np.zeros((n_frames, MAX_PAIRS, 4), dtype=np.float32)  # agent_i, agent_j, prob, ttc
    all_conflict_counts = np.zeros(n_frames, dtype=np.int32)

    t0 = time.time()
    for idx in range(n_frames):
        if idx % 500 == 0:
            elapsed = time.time() - t0
            print(f"  [{idx}/{n_frames}] {elapsed:.0f}s")

        try:
            p = loader.load_frame(idx)
        except Exception:
            continue

        ego = p.ego
        all_ego[idx] = [ego.x, ego.y, ego.vx, ego.vy, ego.heading, ego.velocity, 4.5, 0]

        # Encode agents
        for i, agent in enumerate(p.agents[:MAX_AGENTS]):
            a = agent.state
            all_agents[idx, i] = [a.x, a.y, a.vx, a.vy, a.heading, a.length, a.width, encode_type(a.vehicle_type)]
        all_agent_counts[idx] = min(len(p.agents), MAX_AGENTS)

        # Generate risk map labels: sample query points in ego vicinity
        risk_regions = risk_map_builder.build(p)
        # Sample random points near ego and look up their risk
        rng = np.random.RandomState(idx)
        for qi in range(N_QUERY_POINTS):
            qx = ego.x + rng.uniform(-40, 40)
            qy = ego.y + rng.uniform(-20, 20)
            # Find which risk region this point is in
            risk_val = 0.0
            from shapely.geometry import Point
            pt = Point(qx, qy)
            for region in risk_regions:
                from shapely.geometry import Polygon
                try:
                    if Polygon(region.polygon).contains(pt):
                        risk_val = region.risk_score
                        break
                except:
                    pass
            all_risk_points[idx, qi] = [qx, qy, risk_val]

        # Generate conflict labels
        edges = risk_graph_builder.build(ego, p.agents[:MAX_AGENTS])
        for ei, edge in enumerate(edges[:MAX_PAIRS]):
            # Encode agent indices
            a_idx = int(edge.agent_a_id.split("_")[-1]) if "obj_" in edge.agent_a_id else -1
            b_idx = int(edge.agent_b_id.split("_")[-1]) if "obj_" in edge.agent_b_id else -1
            ttc_val = min(edge.ttc, 20.0)
            all_conflict[idx, ei] = [a_idx, b_idx, edge.collision_probability, ttc_val]
        all_conflict_counts[idx] = min(len(edges), MAX_PAIRS)

    # Save
    np.save(out / "ego_features.npy", all_ego)
    np.save(out / "agent_features.npy", all_agents)
    np.save(out / "agent_counts.npy", all_agent_counts)
    np.save(out / "risk_points.npy", all_risk_points)
    np.save(out / "conflict_labels.npy", all_conflict)
    np.save(out / "conflict_counts.npy", all_conflict_counts)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Saved to {out}")
    print(f"  ego: {all_ego.shape}")
    print(f"  agents: {all_agents.shape}")
    print(f"  risk_points: {all_risk_points.shape}")
    print(f"  conflicts: {all_conflict.shape}")


if __name__ == "__main__":
    main()

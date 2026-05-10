"""Quick DAIR-V2X cross-dataset test for V1 detector.

DAIR-V2X has no collision labels (it's real-world driving, not accident).
Strategy: measure V1 false-alarm rate on DAIR-V2X (assumed all "normal")
to verify V1 doesn't fire on truly normal driving.

Lower V1 prob distribution on DAIR-V2X = better generalization.
"""
from __future__ import annotations
import sys, os, math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.dairv2x_loader_v2 import DAIRv2xLoaderV2


def main():
    print("=" * 60)
    print("  Cross-Dataset Test: V1 on DAIR-V2X")
    print("=" * 60)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    loader = DAIRv2xLoaderV2(
        '/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure')

    # Get a sample of frames
    n_test = min(500, len(loader.coop_info))
    print(f"\n  Testing {n_test} frames...")

    all_probs = []
    for i in range(n_test):
        try:
            perception = loader.load_frame(i)
            if perception is None or not perception.agents:
                continue
            agents = perception.agents
            # Encode for V1
            agents_feat = np.zeros((1, 30, 10), dtype=np.float32)
            mask = np.zeros((1, 30), dtype=bool)
            for j, a in enumerate(agents[:30]):
                s = a.state
                agents_feat[0, j] = [
                    s.x, s.y, s.vx, s.vy, s.heading,
                    s.length, s.width, s.velocity,
                    1.0 if a.is_visible else 0.0, 0]
                mask[0, j] = True
            with torch.no_grad():
                cp, _ = v1(torch.FloatTensor(agents_feat),
                           torch.BoolTensor(mask))
            all_probs.append(cp.item())
        except Exception as e:
            continue

    if not all_probs:
        print("  Failed to compute any probabilities")
        return

    arr = np.array(all_probs)
    print(f"\n  Sample size: {len(arr)}")
    print(f"  Prob distribution: mean={arr.mean():.3f}, median={np.median(arr):.3f}, "
          f"p90={np.percentile(arr,90):.3f}, p99={np.percentile(arr,99):.3f}, "
          f"max={arr.max():.3f}")

    # FA rate at different thresholds
    print(f"\n  FA rate (frame-level) at different thresholds:")
    print(f"  {'Threshold':>10s}  {'FA rate':>10s}")
    print("  " + "-" * 25)
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        fa = (arr > thr).mean()
        print(f"  {thr:9.2f}   {fa:9.1%}")

    # Compare to DeepAccident val frame-level FA at same thresholds
    print("\n  For reference (DeepAccident val, normal frames):")
    print("    thr=0.25 FA=24.2%, thr=0.40 FA=14.4%")
    print("\n  If DAIR-V2X FA < DeepAccident FA: V1 generalizes well")
    print("  If DAIR-V2X FA > DeepAccident FA: V1 over-fires on real-world data")


if __name__ == "__main__":
    main()

"""Analyze signal distributions in accident vs normal scenarios.

Goal: find detection signals that fire in accident but NOT in normal,
to reduce false alarm rate while keeping 100% detection.
"""
import sys, math, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.deepaccident_loader import DeepAccidentLoader

loader = DeepAccidentLoader(split='all')

# Per-scenario statistics
stats = {'accident': [], 'normal': []}

for si, s in enumerate(loader.scenarios):
    tag = 'accident' if s['is_accident'] else 'normal'

    min_dist_any = 999
    max_approach = -999
    has_invisible = False
    n_invisible_agents = 0
    n_approach_positive = 0  # agents with approach_speed > 0
    n_close_approaching = 0  # agents within 10m AND approaching
    max_collision_prob_proxy = 0  # approach_speed / dist as proxy

    for fi in range(len(s['frames'])):
        frame = loader.load_frame(si, fi)
        ego = frame.perception.ego
        ego_speed = max(ego.velocity, 1.0)

        for a in frame.perception.agents:
            st = a.state
            dist = math.sqrt(st.x**2 + st.y**2)

            # Approach speed
            rel_vx = st.vx - ego_speed
            if dist > 0.01:
                approach = -(st.x * rel_vx + st.y * st.vy) / dist
            else:
                approach = 0

            min_dist_any = min(min_dist_any, dist)
            max_approach = max(max_approach, approach)

            if not a.is_visible:
                has_invisible = True
                n_invisible_agents += 1

            if approach > 0:
                n_approach_positive += 1

            if dist < 10 and approach > 2.0:
                n_close_approaching += 1

            if dist > 0.1:
                proxy = approach / dist
                max_collision_prob_proxy = max(max_collision_prob_proxy, proxy)

    stats[tag].append({
        'name': s['name'],
        'min_dist': min_dist_any,
        'max_approach': max_approach,
        'has_invisible': has_invisible,
        'n_invisible': n_invisible_agents,
        'n_approach_pos': n_approach_positive,
        'n_close_approaching': n_close_approaching,
        'collision_proxy': max_collision_prob_proxy,
    })

# Print analysis
for tag in ['accident', 'normal']:
    print(f"\n{'='*60}")
    print(f"  {tag.upper()} scenarios ({len(stats[tag])})")
    print(f"{'='*60}")

    dists = [s['min_dist'] for s in stats[tag]]
    approaches = [s['max_approach'] for s in stats[tag]]
    invis = sum(1 for s in stats[tag] if s['has_invisible'])
    n_invis = [s['n_invisible'] for s in stats[tag]]
    close_app = [s['n_close_approaching'] for s in stats[tag]]
    proxies = [s['collision_proxy'] for s in stats[tag]]

    print(f"  min_dist:     min={min(dists):.1f}, mean={np.mean(dists):.1f}, max={max(dists):.1f}")
    print(f"  max_approach: min={min(approaches):.1f}, mean={np.mean(approaches):.1f}, max={max(approaches):.1f}")
    print(f"  has_invisible: {invis}/{len(stats[tag])} ({invis/len(stats[tag]):.0%})")
    print(f"  n_invisible:  mean={np.mean(n_invis):.1f}, max={max(n_invis)}")
    print(f"  close_approaching (d<10m & v>2m/s): mean={np.mean(close_app):.1f}, max={max(close_app)}")
    print(f"  collision_proxy (approach/dist): mean={np.mean(proxies):.2f}, max={max(proxies):.2f}")

# Test different detection signals
print(f"\n{'='*60}")
print("  DETECTION SIGNAL ANALYSIS")
print(f"{'='*60}")

signals = {
    'has_invisible': lambda s: s['has_invisible'],
    'min_dist < 5m': lambda s: s['min_dist'] < 5,
    'min_dist < 8m': lambda s: s['min_dist'] < 8,
    'min_dist < 10m': lambda s: s['min_dist'] < 10,
    'max_approach > 3': lambda s: s['max_approach'] > 3,
    'max_approach > 5': lambda s: s['max_approach'] > 5,
    'close_approaching > 0': lambda s: s['n_close_approaching'] > 0,
    'close_approaching > 5': lambda s: s['n_close_approaching'] > 5,
    'collision_proxy > 0.5': lambda s: s['collision_proxy'] > 0.5,
    'collision_proxy > 1.0': lambda s: s['collision_proxy'] > 1.0,
    'invisible OR close_approach>0': lambda s: s['has_invisible'] or s['n_close_approaching'] > 0,
    'invisible OR proxy>0.5': lambda s: s['has_invisible'] or s['collision_proxy'] > 0.5,
}

print(f"\n{'Signal':40s} {'Acc Det':>8s} {'Norm FA':>8s}")
print("-" * 60)
for name, fn in signals.items():
    acc_det = sum(1 for s in stats['accident'] if fn(s))
    norm_fa = sum(1 for s in stats['normal'] if fn(s))
    n_acc = len(stats['accident'])
    n_norm = len(stats['normal'])
    print(f"{name:40s} {acc_det}/{n_acc} ({acc_det/n_acc:.0%})  {norm_fa}/{n_norm} ({norm_fa/n_norm:.0%})")

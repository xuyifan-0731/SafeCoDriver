#!/usr/bin/env python3
"""Robust CARLA simulation experiment — handles actor lifecycle carefully.

Connects to an already-running CARLA server, runs scenarios, saves results.
Designed to avoid the C++ segfault in carla 0.9.15 actor destroy.

Start CARLA first:
    Xvfb :99 -screen 0 1024x768x24 &
    VK_ICD_FILENAMES=/tmp/nvidia_icd.json DISPLAY=:99 \
      /raid/xuyifan/jiqiuyu/third_party/carla/CarlaUE4.sh \
      -carla-rpc-port=2000 -RenderOffScreen -quality-level=Low &

Then run:
    cd /raid/xuyifan/jiqiuyu
    conda activate coop-safety
    python experiments/run_carla_robust.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario


def connect():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world


def collect_scenario(client, world, n_vehicles, n_snapshots=10, interval=0.5):
    """Spawn vehicles, collect perception snapshots, cleanup.

    Uses async mode (no synchronous tick) to avoid actor lifecycle issues.
    Uses batch commands for safe cleanup.
    """
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(False)

    # Spawn
    vehicle_bps = bp_lib.filter("vehicle.*")
    vehicles = []
    for i in range(min(n_vehicles, len(spawn_points))):
        bp = vehicle_bps[i % len(vehicle_bps)]
        if bp.has_attribute("is_invincible"):
            bp.set_attribute("is_invincible", "false")
        sp = spawn_points[i]
        v = world.try_spawn_actor(bp, sp)
        if v is not None:
            v.set_autopilot(True, tm.get_port())
            vehicles.append(v)

    if not vehicles:
        return []

    # Wait for traffic to develop — need enough time for autopilot to accelerate
    time.sleep(8.0)

    # Collect snapshots
    ego = vehicles[0]
    frames = []

    for snap in range(n_snapshots):
        time.sleep(interval)
        try:
            et = ego.get_transform()
            ev = ego.get_velocity()
        except Exception:
            continue

        ego_state = VehicleState(
            id="ego",
            x=et.location.x, y=et.location.y,
            heading=np.radians(et.rotation.yaw),
            velocity=max(np.sqrt(ev.x**2 + ev.y**2), 5.0),  # At least 5 m/s for meaningful reachable set
            vx=ev.x if abs(ev.x) > 0.1 else 5.0 * np.cos(np.radians(et.rotation.yaw)),
            vy=ev.y if abs(ev.y) > 0.1 else 5.0 * np.sin(np.radians(et.rotation.yaw)),
            length=max(ego.bounding_box.extent.x * 2, 4.0),
            width=max(ego.bounding_box.extent.y * 2, 1.5),
        )

        agents = []
        for j, v in enumerate(vehicles[1:], 1):
            try:
                vt = v.get_transform()
                vv = v.get_velocity()
            except Exception:
                continue
            dx = vt.location.x - et.location.x
            dy = vt.location.y - et.location.y
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 80:
                continue
            agents.append(Agent(
                state=VehicleState(
                    id=f"v{j}",
                    x=vt.location.x, y=vt.location.y,
                    heading=np.radians(vt.rotation.yaw),
                    velocity=max(np.sqrt(vv.x**2 + vv.y**2), 0.1),
                    vx=vv.x, vy=vv.y,
                    length=max(v.bounding_box.extent.x * 2, 4.0),
                    width=max(v.bounding_box.extent.y * 2, 1.5),
                ),
                agent_type=AgentType.VEHICLE,
            ))

        frames.append(PerceptionResult(
            timestamp=snap * interval,
            ego=ego_state,
            agents=agents,
        ))

    # Safe cleanup via batch command (avoids C++ segfault)
    try:
        ids = [v.id for v in vehicles]
        client.apply_batch([carla.command.DestroyActor(x) for x in ids])
        time.sleep(1.0)
    except Exception as e:
        print(f"  [WARN] Cleanup error (non-fatal): {e}")

    return frames


def main():
    output_dir = Path(__file__).parent / "results" / f"carla_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARLA SIMULATION EXPERIMENT (ROBUST)")
    print("=" * 70)

    client, world = connect()
    print(f"Connected: {world.get_map().name}")

    # Use current map (switching maps causes connection timeout in headless mode)

    # Load methods
    from experiments.methods import get_all_methods
    methods = get_all_methods()
    core = ["NoConstraint", "RSS-Only", "CBF-Based", "Ours-Full", "Ours-NoRiskEvents", "Ours-RiskMapOnly"]
    methods = {k: v for k, v in methods.items() if k in core}
    print(f"Methods: {list(methods.keys())}")

    all_metrics = []
    scenarios_config = [
        (10, "medium"),
        (20, "hard"),
        (30, "hard"),
        (50, "hard"),
    ]

    for n_veh, difficulty in scenarios_config:
        print(f"\n--- Scenario: {n_veh} vehicles ---")

        frames = collect_scenario(client, world, n_veh, n_snapshots=10, interval=0.5)
        print(f"  Collected {len(frames)} frames")

        if not frames:
            print("  [SKIP] No frames collected")
            continue

        for idx, perception in enumerate(frames):
            n_nearby = len(perception.agents)
            config = ScenarioConfig(
                name=f"carla_{n_veh}v_snap{idx}",
                description=f"CARLA {n_veh}v, {n_nearby} nearby agents",
                seed=42 + idx,
                difficulty=difficulty,
            )
            for mname, method in methods.items():
                m = evaluate_method_on_scenario(method, perception, config)
                all_metrics.append(m)

        print(f"  Evaluated: {len(frames)} frames × {len(methods)} methods = {len(frames)*len(methods)} experiments")

    # === SAVE RESULTS ===
    print(f"\nTotal experiments: {len(all_metrics)}")

    results = {
        "experiment_info": {
            "timestamp": datetime.now().isoformat(),
            "simulator": "CARLA 0.9.15 (headless, async mode)",
            "map": world.get_map().name,
            "vehicle_counts": [s[0] for s in scenarios_config],
            "snapshots_per_scenario": 10,
            "total_experiments": len(all_metrics),
        },
        "metrics": [
            {k: (str(v) if isinstance(v, float) and (np.isinf(v) or np.isnan(v)) else v)
             for k, v in asdict(m).items()}
            for m in all_metrics
        ],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("CARLA SIMULATION RESULTS")
    print("=" * 70)
    method_names = sorted(set(m.method_name for m in all_metrics))
    print(f"\n{'Method':25s} {'Avg Area':>10s} {'Tight%':>8s} {'Min TTC':>8s} {'MinHarm':>10s} {'Time':>8s}")
    print("-" * 72)
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        avg_area = np.mean([m.feasible_area for m in mm])
        avg_tight = np.mean([m.constraint_tightening_ratio for m in mm])
        avg_ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh = sum(1 for m in mm if m.mode == "minimum_harm")
        avg_time = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:25s} {avg_area:10.1f} {avg_tight:7.1%} {avg_ttc:8.2f} {mh:>5d}/{len(mm):<4d} {avg_time:7.0f}ms")

    summary_lines = [
        f"# CARLA Simulation Results\n",
        f"Generated: {datetime.now().isoformat()}\n",
        f"Simulator: CARLA 0.9.15, Map: {world.get_map().name}\n",
        f"Total experiments: {len(all_metrics)}\n",
        "",
        f"| Method | Avg Area (m²) | Constraint % | Avg Min TTC | Min-Harm | Avg Time |",
        f"|--------|--------------|-------------|------------|----------|----------|",
    ]
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        a = np.mean([m.feasible_area for m in mm])
        t = np.mean([m.constraint_tightening_ratio for m in mm])
        ttc = np.mean([min(m.min_ttc, 100) for m in mm])
        mh = sum(1 for m in mm if m.mode == "minimum_harm")
        tm = np.mean([m.compute_time_ms for m in mm])
        summary_lines.append(f"| {mname} | {a:.1f} | {t:.1%} | {ttc:.2f}s | {mh}/{len(mm)} | {tm:.0f}ms |")

    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    (output_dir / "reproduction_info.json").write_text(json.dumps({
        "command": "cd /raid/xuyifan/jiqiuyu && python experiments/run_carla_robust.py",
        "prerequisite": "CARLA 0.9.15 server must be running on port 2000",
        "carla_start": "VK_ICD_FILENAMES=/tmp/nvidia_icd.json DISPLAY=:99 "
                       "/raid/xuyifan/jiqiuyu/third_party/carla/CarlaUE4.sh "
                       "-carla-rpc-port=2000 -RenderOffScreen -quality-level=Low",
    }, indent=2))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

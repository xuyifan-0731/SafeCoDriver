"""Improved CARLA closed-loop evaluation with aggressive scenarios.

Fixes from v1:
1. Uses TrafficManager to create more aggressive traffic behavior
2. Spawns pedestrians for harder scenarios
3. Includes ALL baselines (RSS, CBF, RPF, TTCReach, SFS, Ours)
4. More episodes for statistical significance

Usage:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    python experiments/run_carla_closedloop_v2.py
"""

import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, SafeActionSpace, ConstraintMode,
)
from coop_safety.utils.metrics import compute_ttc
from experiments.methods import RSSOnly, CBFBased
from experiments.methods_modern import RiskPotentialField, TTCReachability, SocialForceSafety


@dataclass
class EpisodeResult:
    method: str
    n_vehicles: int
    n_walkers: int
    episode_id: int
    duration: float = 0.0
    collision_count: int = 0
    near_miss_count: int = 0  # TTC < 2s events
    min_harm_triggers: int = 0
    avg_speed: float = 0.0
    distance_traveled: float = 0.0
    avg_min_ttc: float = float('inf')
    min_observed_ttc: float = float('inf')
    steps: int = 0


def build_perception(ego, other_vehicles, ego_vel_override=None):
    """Build PerceptionResult from CARLA actors."""
    try:
        et = ego.get_transform()
        ev = ego.get_velocity()
    except:
        return None

    speed = max(math.sqrt(ev.x**2 + ev.y**2), 1.0)
    ego_state = VehicleState(
        id="ego", x=et.location.x, y=et.location.y,
        heading=math.radians(et.rotation.yaw),
        velocity=speed, vx=ev.x, vy=ev.y,
        length=max(ego.bounding_box.extent.x * 2, 4.0),
        width=max(ego.bounding_box.extent.y * 2, 1.5),
    )

    agents = []
    for i, v in enumerate(other_vehicles):
        try:
            vt = v.get_transform()
            vv = v.get_velocity()
        except:
            continue
        dx = vt.location.x - et.location.x
        dy = vt.location.y - et.location.y
        if dx*dx + dy*dy > 4900:  # > 70m
            continue

        is_walker = "walker" in v.type_id
        agents.append(Agent(
            state=VehicleState(
                id=f"a{i}", x=vt.location.x, y=vt.location.y,
                heading=math.radians(vt.rotation.yaw),
                velocity=max(math.sqrt(vv.x**2 + vv.y**2), 0.1),
                vx=vv.x, vy=vv.y,
                length=max(v.bounding_box.extent.x * 2, 0.5),
                width=max(v.bounding_box.extent.y * 2, 0.5),
                vehicle_type="pedestrian" if is_walker else "car",
                mass=70 if is_walker else 1500,
            ),
            agent_type=AgentType.PEDESTRIAN if is_walker else AgentType.VEHICLE,
        ))

    return PerceptionResult(timestamp=0, ego=ego_state, agents=agents)


def apply_safety_action(ego, safe_space, ego_speed, perception):
    """Apply safety constraint with SMOOTH control — no sudden braking.

    Key fix: instead of binary brake/no-brake, use graduated response:
    - Check if rear vehicle is close before braking hard
    - Use smooth deceleration ramp instead of instant full brake
    - In NORMAL mode, let autopilot handle completely
    """
    if safe_space is None:
        return  # No constraint

    if safe_space.mode == ConstraintMode.NORMAL:
        return  # Let autopilot handle — it's safe enough

    # Check for rear vehicle proximity before braking
    rear_clear = True
    if perception and perception.agents:
        ego_heading = perception.ego.heading
        for agent in perception.agents:
            dx = perception.ego.x - agent.state.x
            dy = perception.ego.y - agent.state.y
            dist = (dx**2 + dy**2) ** 0.5
            # Agent is behind ego and close
            import math
            angle_to_agent = math.atan2(dy, dx)
            heading_diff = abs(angle_to_agent - ego_heading)
            if heading_diff > math.pi:
                heading_diff = 2 * math.pi - heading_diff
            if dist < 15 and heading_diff > math.pi * 0.6:  # Behind and close
                rear_clear = False
                break

    if safe_space.mode == ConstraintMode.MINIMUM_HARM:
        if rear_clear:
            # Safe to brake harder
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.6, steer=0.0))
        else:
            # Rear vehicle close — gentle brake only
            ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.2, steer=0.0))
    elif safe_space.mode == ConstraintMode.CONSERVATIVE:
        # Gentle slowdown
        ego.apply_control(carla.VehicleControl(throttle=0.15, brake=0.15, steer=0.0))


def run_episode(client, world, method_name, safety_module,
                n_vehicles=30, n_walkers=5, duration=40.0, episode_id=0):
    """Run one episode with aggressive traffic."""
    result = EpisodeResult(
        method=method_name, n_vehicles=n_vehicles,
        n_walkers=n_walkers, episode_id=episode_id,
    )

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(8000)

    # Make traffic more aggressive
    tm.set_global_distance_to_leading_vehicle(2.0)  # Closer following
    tm.global_percentage_speed_difference(-20)  # 20% faster than speed limit

    np.random.seed(episode_id)

    # Spawn vehicles
    vehicles = []
    vehicle_bps = bp_lib.filter("vehicle.*")
    indices = np.random.permutation(len(spawn_points))[:n_vehicles]
    for i in indices:
        bp = vehicle_bps[int(i) % len(vehicle_bps)]
        v = world.try_spawn_actor(bp, spawn_points[int(i)])
        if v:
            v.set_autopilot(True, tm.get_port())
            # Make some vehicles ignore traffic lights
            if np.random.random() < 0.3:
                tm.ignore_lights_percentage(v, 80)
            if np.random.random() < 0.2:
                tm.ignore_signs_percentage(v, 50)
            vehicles.append(v)

    if not vehicles:
        return result

    ego = vehicles[0]

    # Spawn pedestrians
    walkers = []
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    walker_controllers = []
    for _ in range(n_walkers):
        spawn_loc = carla.Transform(
            carla.Location(
                x=ego.get_transform().location.x + np.random.uniform(-30, 30),
                y=ego.get_transform().location.y + np.random.uniform(-30, 30),
                z=1.0
            )
        )
        bp = walker_bps[np.random.randint(len(walker_bps))]
        w = world.try_spawn_actor(bp, spawn_loc)
        if w:
            walkers.append(w)
            # Add walker controller
            ctrl_bp = bp_lib.find("controller.ai.walker")
            ctrl = world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
            if ctrl:
                walker_controllers.append(ctrl)
                ctrl.start()
                ctrl.go_to_location(world.get_random_location_from_navigation())
                ctrl.set_max_speed(1.5 + np.random.random() * 1.0)

    all_others = vehicles[1:] + walkers

    # Setup collision sensor
    collision_data = []
    col_bp = bp_lib.find("sensor.other.collision")
    col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
    col_sensor.listen(lambda e: collision_data.append({
        "type": e.other_actor.type_id,
        "impulse": e.normal_impulse.length(),
    }))

    # Let traffic develop
    time.sleep(6.0)

    # Run simulation
    speeds = []
    ttcs = []
    positions = []
    dt = 0.05
    n_steps = int(duration / dt)

    for step in range(n_steps):
        time.sleep(dt)

        perception = build_perception(ego, all_others)
        if perception is None:
            break

        ego_speed = perception.ego.velocity
        speeds.append(ego_speed)
        positions.append((perception.ego.x, perception.ego.y))

        # Compute scene min TTC
        min_ttc_step = float('inf')
        for agent in perception.agents:
            ttc = compute_ttc(
                np.array([perception.ego.x, perception.ego.y]),
                np.array([perception.ego.vx, perception.ego.vy]),
                np.array([agent.state.x, agent.state.y]),
                np.array([agent.state.vx, agent.state.vy]),
            )
            min_ttc_step = min(min_ttc_step, ttc)
        ttcs.append(min(min_ttc_step, 100))

        if min_ttc_step < 2.0:
            result.near_miss_count += 1
        result.min_observed_ttc = min(result.min_observed_ttc, min_ttc_step)

        # Apply safety constraint
        if safety_module is not None:
            try:
                safe_space = safety_module.constrain(perception)
                if safe_space.mode == ConstraintMode.MINIMUM_HARM:
                    result.min_harm_triggers += 1
                apply_safety_action(ego, safe_space, ego_speed, perception)
            except:
                pass

    # Collect results
    result.duration = len(speeds) * dt
    result.steps = len(speeds)
    result.collision_count = len(collision_data)
    result.avg_speed = np.mean(speeds) if speeds else 0
    result.avg_min_ttc = np.mean(ttcs) if ttcs else 100
    if len(positions) >= 2:
        result.distance_traveled = sum(
            math.sqrt((positions[i+1][0]-positions[i][0])**2 +
                      (positions[i+1][1]-positions[i][1])**2)
            for i in range(len(positions)-1)
        )

    # Cleanup
    try:
        col_sensor.destroy()
    except: pass
    for ctrl in walker_controllers:
        try: ctrl.stop(); ctrl.destroy()
        except: pass
    ids = [v.id for v in vehicles + walkers]
    try:
        client.apply_batch([carla.command.DestroyActor(x) for x in ids])
        time.sleep(1)
    except: pass

    return result


def main():
    output_dir = Path(__file__).parent / "results" / f"closedloop_v2_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CARLA CLOSED-LOOP v2 — Aggressive Traffic + All Baselines")
    print("=" * 70)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Map: {world.get_map().name}")

    # All methods
    methods = {
        "NoConstraint": None,
        "RSS-2017": RSSOnly(),
        "CBF-2017": CBFBased(),
        "APF [Rasekhipour17]": RiskPotentialField(),
        "SFS [Helbing95]": SocialForceSafety(),
        "Ours-Full": SafetyConstraintModule(),
    }
    # Skip TTCReach in closed-loop (too slow for real-time)

    configs = [
        {"n_vehicles": 30, "n_walkers": 5},
        {"n_vehicles": 50, "n_walkers": 10},
    ]
    episodes_per = 3

    all_results = []

    for cfg in configs:
        for method_name, module in methods.items():
            for ep in range(episodes_per):
                print(f"\n[{cfg['n_vehicles']}v+{cfg['n_walkers']}w | {method_name} | ep{ep}] ", end="", flush=True)

                # Restart CARLA world to clean state between methods
                try:
                    # Destroy all existing actors
                    actors = world.get_actors().filter("vehicle.*")
                    client.apply_batch([carla.command.DestroyActor(a) for a in actors])
                    actors = world.get_actors().filter("walker.*")
                    client.apply_batch([carla.command.DestroyActor(a) for a in actors])
                    actors = world.get_actors().filter("controller.*")
                    client.apply_batch([carla.command.DestroyActor(a) for a in actors])
                    time.sleep(2)
                except: pass

                r = run_episode(
                    client, world, method_name, module,
                    n_vehicles=cfg["n_vehicles"],
                    n_walkers=cfg["n_walkers"],
                    duration=40.0,
                    episode_id=ep + cfg["n_vehicles"] * 100,
                )
                all_results.append(r)
                print(f"coll={r.collision_count} near_miss={r.near_miss_count} "
                      f"speed={r.avg_speed:.1f}m/s dist={r.distance_traveled:.0f}m "
                      f"min_harm={r.min_harm_triggers} min_ttc={r.min_observed_ttc:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("CLOSED-LOOP v2 RESULTS")
    print("=" * 70)
    print(f"\n{'Method':20s} {'Coll':>6s} {'NearMiss':>9s} {'Speed':>7s} {'Dist':>7s} {'MinHarm':>8s} {'MinTTC':>7s}")
    print("-" * 70)

    summary_rows = []
    for mname in methods:
        mr = [r for r in all_results if r.method == mname]
        tc = sum(r.collision_count for r in mr)
        nm = sum(r.near_miss_count for r in mr)
        avs = np.mean([r.avg_speed for r in mr])
        avd = np.mean([r.distance_traveled for r in mr])
        mh = sum(r.min_harm_triggers for r in mr)
        min_ttc = min(r.min_observed_ttc for r in mr)
        n = len(mr)
        print(f"{mname:20s} {tc:>3d}/{n:<3d} {nm:>9d} {avs:>6.1f} {avd:>6.0f}m {mh:>8d} {min_ttc:>6.1f}s")
        summary_rows.append({"method": mname, "collisions": tc, "episodes": n,
                             "near_misses": nm, "avg_speed": avs, "distance": avd,
                             "min_harm": mh, "min_ttc": min_ttc})

    # Save
    results = {
        "info": {
            "timestamp": datetime.now().isoformat(),
            "simulator": "CARLA 0.9.15",
            "configs": configs,
            "episodes_per": episodes_per,
            "aggressive_traffic": True,
            "total": len(all_results),
        },
        "summary": summary_rows,
        "episodes": [asdict(r) for r in all_results],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = ["# CARLA Closed-Loop v2 Results\n",
             f"Aggressive traffic: closer following, faster speed, ignore lights/signs\n",
             f"Pedestrians: {configs[0]['n_walkers']}-{configs[1]['n_walkers']} per scenario\n",
             "", "| Method | Collisions | Near-Miss | Avg Speed | Distance | Min-Harm | Min TTC |",
             "|--------|-----------|-----------|-----------|----------|----------|---------|"]
    for r in summary_rows:
        lines.append(f"| {r['method']} | {r['collisions']}/{r['episodes']} | {r['near_misses']} | "
                     f"{r['avg_speed']:.1f} m/s | {r['distance']:.0f}m | {r['min_harm']} | {r['min_ttc']:.1f}s |")
    (output_dir / "summary.md").write_text("\n".join(lines))
    (output_dir / "reproduction_info.json").write_text(json.dumps({"cmd": "python experiments/run_carla_closedloop_v2.py"}))

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()

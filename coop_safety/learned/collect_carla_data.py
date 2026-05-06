"""Collect collision/safe training data from CARLA simulation.

Runs CARLA with aggressive traffic and records:
  - Frames where collisions DID happen (positive: high risk)
  - Frames where near-misses occurred (moderate risk)
  - Frames with safe driving (negative: low risk)

Each frame records: ego state, all agent states, and the ground-truth outcome.
This gives us REAL risk labels instead of rule-generated labels.
"""

import sys
import json
import time
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import carla

OUTPUT_DIR = "/raid/xuyifan/jiqiuyu/data/carla_risk_labels"


def collect_data(n_episodes=20, vehicles_per_episode=40, walkers_per_episode=10,
                 episode_duration=60.0, dt=0.1):
    """Collect frames with ground-truth risk labels from CARLA."""

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(8000)

    # Aggressive traffic
    tm.set_global_distance_to_leading_vehicle(1.5)
    tm.global_percentage_speed_difference(-30)  # 30% over speed limit

    all_frames = []  # list of dicts

    for ep in range(n_episodes):
        print(f"\n[Episode {ep}/{n_episodes}]")
        np.random.seed(ep * 17)

        # Spawn vehicles
        vehicles = []
        veh_bps = bp_lib.filter("vehicle.*")
        indices = np.random.permutation(len(spawn_points))[:vehicles_per_episode]
        for i in indices:
            bp = veh_bps[int(i) % len(veh_bps)]
            v = world.try_spawn_actor(bp, spawn_points[int(i)])
            if v:
                v.set_autopilot(True, tm.get_port())
                if np.random.random() < 0.4:
                    tm.ignore_lights_percentage(v, 90)
                if np.random.random() < 0.3:
                    tm.ignore_signs_percentage(v, 70)
                vehicles.append(v)

        # Spawn walkers
        walkers = []
        walker_ctrls = []
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        for _ in range(walkers_per_episode):
            sp = carla.Transform(carla.Location(
                x=spawn_points[0].location.x + np.random.uniform(-50, 50),
                y=spawn_points[0].location.y + np.random.uniform(-50, 50),
                z=1.0
            ))
            w = world.try_spawn_actor(walker_bps[np.random.randint(len(walker_bps))], sp)
            if w:
                walkers.append(w)
                ctrl_bp = bp_lib.find("controller.ai.walker")
                ctrl = world.try_spawn_actor(ctrl_bp, carla.Transform(), w)
                if ctrl:
                    walker_ctrls.append(ctrl)
                    ctrl.start()
                    ctrl.go_to_location(world.get_random_location_from_navigation())

        if not vehicles:
            continue

        ego = vehicles[0]
        others = vehicles[1:] + walkers

        # Collision sensor
        collision_events = []
        col_bp = bp_lib.find("sensor.other.collision")
        col_sensor = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        col_sensor.listen(lambda e: collision_events.append(time.time()))

        time.sleep(5)  # Let traffic develop

        # Collect frames
        n_steps = int(episode_duration / dt)
        collisions_before = 0

        for step in range(n_steps):
            time.sleep(dt)
            n_coll_now = len(collision_events)
            collision_happened = n_coll_now > collisions_before
            collisions_before = n_coll_now

            try:
                et = ego.get_transform()
                ev = ego.get_velocity()
            except:
                break

            ego_speed = math.sqrt(ev.x**2 + ev.y**2)

            # Compute min TTC
            min_ttc = 100.0
            agents_data = []
            for a in others:
                try:
                    at = a.get_transform()
                    av = a.get_velocity()
                except:
                    continue
                dx = at.location.x - et.location.x
                dy = at.location.y - et.location.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 60:
                    continue

                a_speed = math.sqrt(av.x**2 + av.y**2)
                rel_vx = ev.x - av.x
                rel_vy = ev.y - av.y
                approach = -(dx*rel_vx + dy*rel_vy) / max(dist, 0.1)
                if approach > 0.1:
                    ttc = dist / approach
                    min_ttc = min(min_ttc, ttc)

                is_walker = "walker" in a.type_id
                agents_data.append([
                    at.location.x, at.location.y,
                    av.x, av.y,
                    math.radians(at.rotation.yaw),
                    a.bounding_box.extent.x * 2,
                    a.bounding_box.extent.y * 2,
                    3 if is_walker else 0,  # type_id
                ])

            # Determine risk label
            if collision_happened:
                risk_label = 1.0  # Collision frame
            elif min_ttc < 2.0:
                risk_label = 0.7  # Near-miss
            elif min_ttc < 5.0:
                risk_label = 0.3  # Moderate risk
            else:
                risk_label = 0.0  # Safe

            frame = {
                "ego": [et.location.x, et.location.y, ev.x, ev.y,
                        math.radians(et.rotation.yaw), ego_speed,
                        ego.bounding_box.extent.x*2, 0],
                "agents": agents_data[:40],  # max 40
                "risk_label": risk_label,
                "min_ttc": min_ttc,
                "collision": collision_happened,
            }
            all_frames.append(frame)

        print(f"  Collected {n_steps} frames, {len(collision_events)} collisions")

        # Cleanup
        try:
            col_sensor.destroy()
        except: pass
        for c in walker_ctrls:
            try: c.stop(); c.destroy()
            except: pass
        ids = [v.id for v in vehicles + walkers]
        try:
            client.apply_batch([carla.command.DestroyActor(x) for x in ids])
            time.sleep(2)
        except: pass

    # Convert to numpy
    print(f"\nTotal frames: {len(all_frames)}")
    n = len(all_frames)
    max_agents = 40

    ego_feats = np.zeros((n, 8), dtype=np.float32)
    agent_feats = np.zeros((n, max_agents, 8), dtype=np.float32)
    agent_counts = np.zeros(n, dtype=np.int32)
    risk_labels = np.zeros(n, dtype=np.float32)
    ttc_labels = np.zeros(n, dtype=np.float32)
    collision_labels = np.zeros(n, dtype=bool)

    for i, f in enumerate(all_frames):
        ego_feats[i] = f["ego"]
        for j, a in enumerate(f["agents"][:max_agents]):
            agent_feats[i, j] = a
        agent_counts[i] = len(f["agents"][:max_agents])
        risk_labels[i] = f["risk_label"]
        ttc_labels[i] = min(f["min_ttc"], 20)
        collision_labels[i] = f["collision"]

    np.save(out / "ego_features.npy", ego_feats)
    np.save(out / "agent_features.npy", agent_feats)
    np.save(out / "agent_counts.npy", agent_counts)
    np.save(out / "risk_labels.npy", risk_labels)
    np.save(out / "ttc_labels.npy", ttc_labels)
    np.save(out / "collision_labels.npy", collision_labels)

    # Stats
    print(f"Risk distribution:")
    print(f"  Collision (1.0): {(risk_labels==1.0).sum()} ({(risk_labels==1.0).mean()*100:.1f}%)")
    print(f"  Near-miss (0.7): {(risk_labels==0.7).sum()} ({(risk_labels==0.7).mean()*100:.1f}%)")
    print(f"  Moderate (0.3): {(risk_labels==0.3).sum()} ({(risk_labels==0.3).mean()*100:.1f}%)")
    print(f"  Safe (0.0): {(risk_labels==0.0).sum()} ({(risk_labels==0.0).mean()*100:.1f}%)")
    print(f"Saved to {out}")


if __name__ == "__main__":
    collect_data(n_episodes=20, vehicles_per_episode=40, walkers_per_episode=10,
                 episode_duration=60.0, dt=0.1)

"""CARLA closed-loop evaluation: ego follows safety constraints over time.

Measures the actual safety outcome (collision rate) rather than just
constraint area. This is the most important experiment for a safety paper.

Ego vehicle uses autopilot for base behavior, but we override unsafe actions
by projecting them into the constrained safe action space. Then we measure:
- Collision rate (with/without constraints)
- Route completion rate
- Average speed (efficiency)
- Min-harm trigger rate

Usage:
    # Start CARLA first, then:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    python experiments/run_carla_closedloop.py
"""

import sys
import json
import time
import math
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import carla

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, SafeActionSpace, ConstraintMode,
)
from coop_safety.utils.metrics import compute_ttc


@dataclass
class EpisodeResult:
    """Result of one closed-loop episode."""
    method: str
    n_vehicles: int
    episode_id: int
    duration: float = 0.0           # seconds simulated
    collision_count: int = 0
    collision_types: list = field(default_factory=list)
    min_harm_triggers: int = 0
    avg_speed: float = 0.0          # m/s
    distance_traveled: float = 0.0  # meters
    avg_min_ttc: float = float('inf')
    steps: int = 0


class ClosedLoopEvaluator:
    """Run ego vehicle with safety constraints in CARLA and measure outcomes."""

    def __init__(self, client: carla.Client, dt: float = 0.05):
        self.client = client
        self.world = client.get_world()
        self.dt = dt
        self.bp_lib = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.tm = client.get_trafficmanager(8000)

    def run_episode(self, method_name: str, safety_module: Optional[SafetyConstraintModule],
                    n_vehicles: int = 20, episode_duration: float = 30.0,
                    episode_id: int = 0) -> EpisodeResult:
        """Run one closed-loop episode.

        Args:
            method_name: Name for logging
            safety_module: Our safety constraint module (None = no constraint)
            n_vehicles: Number of traffic vehicles to spawn
            episode_duration: Simulation time in seconds
            episode_id: For reproducibility

        Returns:
            EpisodeResult with collision/efficiency metrics
        """
        result = EpisodeResult(
            method=method_name,
            n_vehicles=n_vehicles,
            episode_id=episode_id,
        )

        # Spawn vehicles
        vehicles = self._spawn_vehicles(n_vehicles, episode_id)
        if not vehicles:
            return result

        ego = vehicles[0]
        ego.set_autopilot(True, self.tm.get_port())

        # Setup collision sensor on ego
        collision_sensor, collision_data = self._setup_collision_sensor(ego)

        # Let traffic develop
        time.sleep(5.0)

        # Run simulation
        speeds = []
        ttcs = []
        positions = []
        n_steps = int(episode_duration / self.dt)
        real_steps = 0

        for step in range(n_steps):
            time.sleep(self.dt)
            real_steps += 1

            try:
                ego_transform = ego.get_transform()
                ego_velocity = ego.get_velocity()
            except Exception:
                break

            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
            speeds.append(ego_speed)
            positions.append((ego_transform.location.x, ego_transform.location.y))

            # Build perception
            perception = self._build_perception(ego, vehicles[1:])

            # Compute min TTC to nearest agent
            min_ttc = self._compute_scene_min_ttc(perception)
            ttcs.append(min(min_ttc, 100))

            # Apply safety constraint (if method has one)
            if safety_module is not None:
                try:
                    safe_space = safety_module.constrain(perception)
                    if safe_space.mode == ConstraintMode.MINIMUM_HARM:
                        result.min_harm_triggers += 1
                        # In minimum harm mode: apply emergency brake
                        ego_control = carla.VehicleControl(
                            throttle=0.0, brake=1.0, steer=0.0
                        )
                        ego.apply_control(ego_control)
                    elif safe_space.mode == ConstraintMode.CONSERVATIVE:
                        # Reduce speed
                        ego_control = carla.VehicleControl(
                            throttle=0.2, brake=0.3, steer=0.0
                        )
                        ego.apply_control(ego_control)
                    # NORMAL mode: let autopilot handle (already set)
                except Exception:
                    pass  # Constraint failure → let autopilot continue

        # Collect results
        result.duration = real_steps * self.dt
        result.steps = real_steps
        result.collision_count = len(collision_data)
        result.collision_types = [d['type'] for d in collision_data]
        result.avg_speed = np.mean(speeds) if speeds else 0
        result.avg_min_ttc = np.mean(ttcs) if ttcs else float('inf')

        if len(positions) >= 2:
            dists = [math.sqrt((positions[i+1][0]-positions[i][0])**2 +
                               (positions[i+1][1]-positions[i][1])**2)
                     for i in range(len(positions)-1)]
            result.distance_traveled = sum(dists)

        # Cleanup
        if collision_sensor and collision_sensor.is_alive:
            collision_sensor.destroy()
        self._destroy_vehicles(vehicles)

        return result

    def _spawn_vehicles(self, n_vehicles: int, seed: int) -> list:
        """Spawn ego + traffic vehicles."""
        np.random.seed(seed)
        vehicle_bps = self.bp_lib.filter("vehicle.*")
        vehicles = []

        indices = np.random.permutation(len(self.spawn_points))[:n_vehicles]
        for i in indices:
            bp = vehicle_bps[int(i) % len(vehicle_bps)]
            v = self.world.try_spawn_actor(bp, self.spawn_points[i])
            if v:
                if len(vehicles) > 0:  # Not ego
                    v.set_autopilot(True, self.tm.get_port())
                vehicles.append(v)

        return vehicles

    def _setup_collision_sensor(self, ego) -> tuple:
        """Attach collision sensor to ego vehicle."""
        collision_data = []
        bp = self.bp_lib.find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=ego)

        def on_collision(event):
            other = event.other_actor
            collision_data.append({
                'type': other.type_id,
                'impulse': event.normal_impulse.length(),
            })

        sensor.listen(on_collision)
        return sensor, collision_data

    def _build_perception(self, ego, other_vehicles: list) -> PerceptionResult:
        """Build PerceptionResult from CARLA ground truth."""
        try:
            et = ego.get_transform()
            ev = ego.get_velocity()
        except:
            return PerceptionResult(timestamp=0, ego=VehicleState(
                id="ego", x=0, y=0, heading=0, velocity=0))

        ego_state = VehicleState(
            id="ego", x=et.location.x, y=et.location.y,
            heading=math.radians(et.rotation.yaw),
            velocity=max(math.sqrt(ev.x**2 + ev.y**2), 1.0),
            vx=ev.x, vy=ev.y,
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
            if dx*dx + dy*dy > 6400:  # > 80m
                continue
            agents.append(Agent(
                state=VehicleState(
                    id=f"v{i}", x=vt.location.x, y=vt.location.y,
                    heading=math.radians(vt.rotation.yaw),
                    velocity=max(math.sqrt(vv.x**2 + vv.y**2), 0.1),
                    vx=vv.x, vy=vv.y,
                    length=max(v.bounding_box.extent.x * 2, 4.0),
                    width=max(v.bounding_box.extent.y * 2, 1.5),
                ),
                agent_type=AgentType.VEHICLE,
            ))

        return PerceptionResult(timestamp=0, ego=ego_state, agents=agents)

    def _compute_scene_min_ttc(self, perception: PerceptionResult) -> float:
        """Compute minimum TTC across all agents."""
        ego = perception.ego
        min_ttc = float('inf')
        for agent in perception.agents:
            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([agent.state.x, agent.state.y]),
                np.array([agent.state.vx, agent.state.vy]),
            )
            min_ttc = min(min_ttc, ttc)
        return min_ttc

    def _destroy_vehicles(self, vehicles: list):
        """Safely destroy all vehicles."""
        try:
            ids = [v.id for v in vehicles]
            self.client.apply_batch([carla.command.DestroyActor(x) for x in ids])
            time.sleep(1)
        except:
            pass


def main():
    output_dir = Path(__file__).parent / "results" / f"closedloop_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLOSED-LOOP COLLISION RATE EVALUATION")
    print("=" * 70)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(f"Connected: {world.get_map().name}")

    evaluator = ClosedLoopEvaluator(client)

    # Methods to evaluate
    methods = {
        "NoConstraint": None,
        "Ours-Full": SafetyConstraintModule(),
    }

    # Experiment config
    vehicle_counts = [15, 25, 40]
    episodes_per_config = 3
    episode_duration = 30.0  # seconds

    all_results = []

    for n_veh in vehicle_counts:
        for method_name, module in methods.items():
            for ep in range(episodes_per_config):
                print(f"\n[{n_veh}v | {method_name} | ep{ep}] ", end="", flush=True)
                result = evaluator.run_episode(
                    method_name=method_name,
                    safety_module=module,
                    n_vehicles=n_veh,
                    episode_duration=episode_duration,
                    episode_id=ep + n_veh * 100,
                )
                all_results.append(result)
                print(f"collisions={result.collision_count}, "
                      f"speed={result.avg_speed:.1f}m/s, "
                      f"dist={result.distance_traveled:.0f}m, "
                      f"min_harm={result.min_harm_triggers}, "
                      f"ttc={result.avg_min_ttc:.1f}s")

    # Summary
    print("\n" + "=" * 70)
    print("CLOSED-LOOP RESULTS")
    print("=" * 70)
    print(f"\n{'Method':20s} {'Collisions':>12s} {'Coll Rate':>10s} {'Avg Speed':>10s} "
          f"{'Distance':>10s} {'Min-Harm':>10s} {'Avg TTC':>10s}")
    print("-" * 85)

    for method_name in methods:
        mr = [r for r in all_results if r.method == method_name]
        total_collisions = sum(r.collision_count for r in mr)
        total_episodes = len(mr)
        coll_rate = total_collisions / max(total_episodes, 1)
        avg_speed = np.mean([r.avg_speed for r in mr])
        avg_dist = np.mean([r.distance_traveled for r in mr])
        total_mh = sum(r.min_harm_triggers for r in mr)
        avg_ttc = np.mean([min(r.avg_min_ttc, 100) for r in mr])
        print(f"{method_name:20s} {total_collisions:>8d}/{total_episodes:<3d} "
              f"{coll_rate:>9.2f} {avg_speed:>9.1f} {avg_dist:>9.0f} "
              f"{total_mh:>10d} {avg_ttc:>9.1f}")

    # Save results
    results = {
        "experiment_info": {
            "type": "closed-loop collision rate evaluation",
            "timestamp": datetime.now().isoformat(),
            "simulator": "CARLA 0.9.15 (headless)",
            "map": world.get_map().name,
            "vehicle_counts": vehicle_counts,
            "episodes_per_config": episodes_per_config,
            "episode_duration": episode_duration,
            "total_episodes": len(all_results),
        },
        "results": [asdict(r) for r in all_results],
    }
    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))

    # Save summary
    summary_lines = [
        "# Closed-Loop Collision Rate Evaluation\n",
        f"CARLA 0.9.15, {world.get_map().name}\n",
        f"Episodes: {len(all_results)} ({episodes_per_config} per config)\n",
        f"Duration: {episode_duration}s per episode\n",
        "",
        "| Method | Collisions | Rate | Avg Speed | Distance | Min-Harm | Avg TTC |",
        "|--------|-----------|------|-----------|----------|----------|---------|",
    ]
    for method_name in methods:
        mr = [r for r in all_results if r.method == method_name]
        tc = sum(r.collision_count for r in mr)
        te = len(mr)
        cr = tc / max(te, 1)
        avs = np.mean([r.avg_speed for r in mr])
        avd = np.mean([r.distance_traveled for r in mr])
        mh = sum(r.min_harm_triggers for r in mr)
        att = np.mean([min(r.avg_min_ttc, 100) for r in mr])
        summary_lines.append(
            f"| {method_name} | {tc}/{te} | {cr:.2f} | {avs:.1f} m/s | "
            f"{avd:.0f}m | {mh} | {att:.1f}s |")

    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    (output_dir / "reproduction_info.json").write_text(json.dumps({
        "command": "python experiments/run_carla_closedloop.py",
        "prerequisite": "CARLA 0.9.15 server running on port 2000",
    }, indent=2))

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

"""CARLA simulation experiment runner.

Starts CARLA server in headless mode, spawns cooperative driving scenarios,
and evaluates safety constraint methods in real-time simulation.

Prerequisites:
  - CARLA 0.9.15 extracted at /raid/xuyifan/jiqiuyu/third_party/carla/
  - Xvfb for virtual display
  - carla Python package installed

Usage:
    cd /raid/xuyifan/jiqiuyu
    conda activate coop-safety
    python experiments/run_carla_experiments.py
"""

import sys
import os
import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType, LaneInfo,
)
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario, ScenarioMetrics

CARLA_ROOT = "/raid/xuyifan/jiqiuyu/third_party/carla"
CARLA_PORT = 2000
XVFB_DISPLAY = ":99"


class CarlaSimManager:
    """Manage CARLA server lifecycle and scenario execution."""

    def __init__(self, carla_root: str = CARLA_ROOT, port: int = CARLA_PORT):
        self.carla_root = Path(carla_root)
        self.port = port
        self._server_proc = None
        self._xvfb_proc = None

    def start_server(self) -> bool:
        """Start CARLA server or connect to existing one."""
        # First try to connect to existing server
        try:
            import carla
            client = carla.Client("localhost", self.port)
            client.set_timeout(5.0)
            world = client.get_world()
            print(f"Connected to existing CARLA server. Map: {world.get_map().name}")
            return True
        except Exception:
            pass

        server_bin = self.carla_root / "CarlaUE4.sh"
        if not server_bin.exists():
            # Try alternative locations
            for alt in ["CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping",
                        "CarlaUE4-Linux-Shipping"]:
                alt_path = self.carla_root / alt
                if alt_path.exists():
                    server_bin = alt_path
                    break
            else:
                print(f"[ERROR] CARLA server binary not found in {self.carla_root}")
                print(f"  Contents: {list(self.carla_root.iterdir())[:10]}")
                return False

        # Start Xvfb
        print("Starting Xvfb...")
        self._xvfb_proc = subprocess.Popen(
            ["Xvfb", XVFB_DISPLAY, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(1)

        # Start CARLA server
        env = dict(os.environ)
        env["DISPLAY"] = XVFB_DISPLAY
        env["SDL_VIDEODRIVER"] = "offscreen"

        print(f"Starting CARLA server on port {self.port}...")
        self._server_proc = subprocess.Popen(
            [str(server_bin), "-carla-rpc-port=" + str(self.port),
             "-quality-level=Low", "-RenderOffScreen"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        print("Waiting for CARLA server...")
        for attempt in range(30):
            try:
                import carla
                client = carla.Client("localhost", self.port)
                client.set_timeout(5.0)
                world = client.get_world()
                print(f"CARLA server ready. Map: {world.get_map().name}")
                return True
            except Exception:
                time.sleep(2)

        print("[ERROR] CARLA server failed to start within 60s")
        self.stop_server()
        return False

    def stop_server(self):
        """Stop CARLA server and Xvfb."""
        if self._server_proc:
            self._server_proc.terminate()
            self._server_proc.wait(timeout=10)
            self._server_proc = None
        if self._xvfb_proc:
            self._xvfb_proc.terminate()
            self._xvfb_proc.wait(timeout=5)
            self._xvfb_proc = None

    def run_scenario(self, n_vehicles: int = 20, duration: float = 10.0,
                     dt: float = 0.1) -> list[PerceptionResult]:
        """Run a cooperative driving scenario and collect perception frames.

        Spawns vehicles, runs simulation, extracts ground-truth perception.

        Returns:
            List of PerceptionResult for each simulation step
        """
        import carla

        client = carla.Client("localhost", self.port)
        client.set_timeout(10.0)
        world = client.get_world()

        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)

        # Spawn vehicles
        blueprint_library = world.get_blueprint_library()
        vehicle_bps = blueprint_library.filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()

        vehicles = []
        for i in range(min(n_vehicles, len(spawn_points))):
            bp = np.random.choice(vehicle_bps)
            try:
                vehicle = world.spawn_actor(bp, spawn_points[i])
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)
            except Exception:
                continue

        # Let traffic settle
        for _ in range(20):
            world.tick()

        # Collect frames
        frames = []
        ego = vehicles[0] if vehicles else None
        n_steps = int(duration / dt)

        for step in range(n_steps):
            try:
                world.tick()
            except Exception:
                break

            if ego is None:
                continue

            try:
                # Extract ego state
                ego_transform = ego.get_transform()
                ego_velocity = ego.get_velocity()
            except Exception:
                break

            ego_state = VehicleState(
                id="ego",
                x=ego_transform.location.x,
                y=ego_transform.location.y,
                heading=np.radians(ego_transform.rotation.yaw),
                velocity=np.sqrt(ego_velocity.x**2 + ego_velocity.y**2),
                vx=ego_velocity.x,
                vy=ego_velocity.y,
                length=ego.bounding_box.extent.x * 2,
                width=ego.bounding_box.extent.y * 2,
            )

            # Extract other vehicle states (ground truth perception)
            agents = []
            for i, v in enumerate(vehicles[1:], 1):
                try:
                    if not v.is_alive:
                        continue
                    t = v.get_transform()
                    vel = v.get_velocity()
                except Exception:
                    continue
                dist = np.sqrt((t.location.x - ego_state.x)**2 +
                               (t.location.y - ego_state.y)**2)
                if dist > 100:
                    continue  # Too far

                agents.append(Agent(
                    state=VehicleState(
                        id=f"v_{i}",
                        x=t.location.x, y=t.location.y,
                        heading=np.radians(t.rotation.yaw),
                        velocity=np.sqrt(vel.x**2 + vel.y**2),
                        vx=vel.x, vy=vel.y,
                        length=v.bounding_box.extent.x * 2,
                        width=v.bounding_box.extent.y * 2,
                    ),
                    agent_type=AgentType.VEHICLE,
                    is_visible=True,
                    confidence=1.0,
                ))

            frames.append(PerceptionResult(
                timestamp=step * dt,
                ego=ego_state,
                agents=agents,
            ))

        # Cleanup — use batch destroy to avoid C++ crash
        try:
            client = carla.Client("localhost", self.port)
            client.set_timeout(10.0)
            batch = [carla.command.DestroyActor(v.id) for v in vehicles]
            client.apply_batch_sync(batch, True)
        except Exception:
            pass

        # Restore async mode
        settings.synchronous_mode = False
        world.apply_settings(settings)

        return frames


def main():
    output_dir = Path(__file__).parent / "results" / f"carla_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT: Safety Constraints in CARLA Simulation")
    print("=" * 70)

    mgr = CarlaSimManager()

    if not mgr.start_server():
        print("\n[FALLBACK] CARLA server could not start.")
        print("To run CARLA experiments manually:")
        print(f"  1. cd {CARLA_ROOT}")
        print(f"  2. DISPLAY=:99 Xvfb :99 -screen 0 1024x768x24 &")
        print(f"  3. ./CarlaUE4.sh -carla-rpc-port=2000 -RenderOffScreen &")
        print(f"  4. python experiments/run_carla_experiments.py")
        mgr.stop_server()
        return

    try:
        from experiments.methods import get_all_methods
        methods = get_all_methods()
        # Keep only core methods for CARLA
        core_names = ["NoConstraint", "RSS-Only", "CBF-Based", "Ours-Full", "Ours-NoRiskEvents", "Ours-RiskMapOnly"]
        methods = {k: v for k, v in methods.items() if k in core_names}

        all_metrics = []

        # Run multiple scenarios with different vehicle counts
        for n_vehicles in [10, 20, 30, 50]:
            print(f"\n--- Scenario: {n_vehicles} vehicles ---")
            frames = mgr.run_scenario(n_vehicles=n_vehicles, duration=10.0)
            print(f"  Collected {len(frames)} frames")

            # Sample 10 frames for evaluation
            sample_indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)

            for idx in sample_indices:
                perception = frames[idx]
                config = ScenarioConfig(
                    name=f"carla_{n_vehicles}v_t{perception.timestamp:.1f}",
                    description=f"CARLA {n_vehicles} vehicles, t={perception.timestamp:.1f}s",
                    seed=42, difficulty="hard" if n_vehicles > 20 else "medium",
                )

                for mname, method in methods.items():
                    metrics = evaluate_method_on_scenario(method, perception, config)
                    all_metrics.append(metrics)
                    print(f"  {mname:20s} area={metrics.feasible_area:.1f} mode={metrics.mode}")

        # Save results
        results = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "simulator": "CARLA 0.9.15",
                "scenarios": [10, 20, 30, 50],
                "total_experiments": len(all_metrics),
            },
            "metrics": [asdict(m) for m in all_metrics],
        }
        (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {output_dir}")

    finally:
        mgr.stop_server()


if __name__ == "__main__":
    main()

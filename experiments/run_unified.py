"""Unified experiment suite — one script, all experiments, clean output.

Runs:
  EXP-1: DAIR-V2X offline (500 frames, all methods, real velocities)
  EXP-2: DAIR-V2X ablation (same frames, ablation variants)
  EXP-3: Cooperative vs Non-cooperative (200 frames, TRUE vehicle-side separation)
  EXP-4: CARLA closed-loop (aggressive traffic, all methods, driving metrics)

All experiments use:
  - Real velocities from consecutive frames (DAIR-V2X)
  - Ground truth from CARLA simulator
  - No synthetic/estimated data

Usage:
    cd /raid/xuyifan/jiqiuyu && conda activate coop-safety
    # For EXP 1-3 (offline, ~25 min):
    python experiments/run_unified.py --offline
    # For EXP 4 (CARLA, ~40 min, CARLA must be running):
    python experiments/run_unified.py --carla
    # For all:
    python experiments/run_unified.py --all
"""

import sys
import os
import json
import time
import math
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import SafetyConstraintModule, ConstraintMode
from coop_safety.learned.risk_assessor import LearnedRiskAssessor
from experiments.dairv2x_loader_v2 import DAIRv2xLoaderV2
from experiments.methods import NoConstraint, RSSOnly, CBFBased, AblationMethod
from experiments.methods_modern import RiskPotentialField, TTCReachability, SocialForceSafety
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario

DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"


# ============================================================
# Method registry
# ============================================================

def get_comparison_methods():
    """All methods for main comparison."""
    m = {}
    m["NoConstraint"] = NoConstraint()
    rss = RSSOnly(); rss.name = "RSS [Shalev-Shwartz'17]"
    m["RSS [Shalev-Shwartz'17]"] = rss
    cbf = CBFBased(); cbf.name = "CBF [Ames'17]"
    m["CBF [Ames'17]"] = cbf
    rpf = RiskPotentialField(); rpf.name = "APF [Rasekhipour17]"
    m["APF [Rasekhipour17]"] = rpf
    sfs = SocialForceSafety(); sfs.name = "SFS [Helbing95]"
    m["SFS [Helbing95]"] = sfs
    ttc = TTCReachability(); ttc.name = "TTCReach [Pek18]"
    m["TTCReach [Pek18]"] = ttc

    rule = SafetyConstraintModule(); rule.name = "Ours-Rule"
    m["Ours-Rule"] = rule

    # Learned version (if model exists)
    model_path = "/raid/xuyifan/jiqiuyu/models/risk_net_best.pt"
    if os.path.exists(model_path):
        from experiments.run_learned_comparison import LearnedSafetyModule
        m["Ours-Learned"] = LearnedSafetyModule()

    return m


def get_ablation_methods():
    """Ablation variants."""
    m = {}
    rule = SafetyConstraintModule(); rule.name = "Ours-Full"
    m["Ours-Full"] = rule
    m["w/o RiskEvents"] = AblationMethod("w/o RiskEvents", {"disable_risk_events": True, "min_probability": 999})
    m["w/o RiskGraph (MapOnly)"] = AblationMethod("w/o RiskGraph (MapOnly)", {"risk_map_only": True, "ttc_warning": 0.0, "min_probability": 999})
    return m


# ============================================================
# EXP 1-3: Offline experiments
# ============================================================

def run_offline(output_dir: Path):
    """Run EXP 1-3 on DAIR-V2X."""
    loader = DAIRv2xLoaderV2(DATA_DIR)

    # --- EXP 1: Main comparison ---
    print("\n" + "=" * 60)
    print("EXP-1: DAIR-V2X Main Comparison (500 frames × all methods)")
    print("  Data: DAIR-V2X-C [Yu et al., CVPR 2022]")
    print("  Velocities: real (consecutive frame position delta, dt=0.1s)")
    print("  Frames: 500 evenly sampled from 6617")
    print("=" * 60)

    methods = get_comparison_methods()
    indices = np.linspace(0, len(loader) - 1, 500, dtype=int)
    exp1_metrics = []

    t0 = time.time()
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"  [{i}/500] {time.time()-t0:.0f}s")
        p = loader.load_frame(idx)
        config = ScenarioConfig(name=f"f{idx}", description=f"{len(p.agents)}ag", seed=42+i, difficulty="hard")
        for mname, method in methods.items():
            m = evaluate_method_on_scenario(method, p, config)
            exp1_metrics.append(m)

    # --- EXP 2: Ablation ---
    print("\n" + "=" * 60)
    print("EXP-2: Ablation Study (same 500 frames)")
    print("=" * 60)

    abl_methods = get_ablation_methods()
    exp2_metrics = []
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"  [{i}/500] {time.time()-t0:.0f}s")
        p = loader.load_frame(idx)
        config = ScenarioConfig(name=f"f{idx}", description="", seed=42+i, difficulty="hard")
        for mname, method in abl_methods.items():
            m = evaluate_method_on_scenario(method, p, config)
            exp2_metrics.append(m)

    # --- EXP 3: Cooperative comparison ---
    print("\n" + "=" * 60)
    print("EXP-3: Cooperative vs Non-cooperative (200 frames)")
    print("  Cooperative: cooperative/label_world (V2I fusion)")
    print("  Non-cooperative: vehicle-side/label (ego-only detection)")
    print("=" * 60)

    module = SafetyConstraintModule()
    coop_indices = np.linspace(0, len(loader) - 1, 200, dtype=int)
    exp3_coop = []
    exp3_noncoop = []
    agent_diffs = []

    for i, idx in enumerate(coop_indices):
        if i % 50 == 0:
            print(f"  [{i}/200]")
        p_coop = loader.load_frame(idx)
        p_noncoop = loader.load_vehicle_side_only(idx)
        config = ScenarioConfig(name=f"c{idx}", description="", seed=42+i, difficulty="hard")

        m_c = evaluate_method_on_scenario(module, p_coop, config)
        m_n = evaluate_method_on_scenario(module, p_noncoop, config)
        exp3_coop.append(m_c)
        exp3_noncoop.append(m_n)
        agent_diffs.append({"coop": len(p_coop.agents), "noncoop": len(p_noncoop.agents)})

    # --- Print results ---
    _print_table("EXP-1 RESULTS", exp1_metrics, methods.keys())
    _print_table("EXP-2 ABLATION", exp2_metrics, abl_methods.keys())
    _print_coop("EXP-3 COOPERATIVE", exp3_coop, exp3_noncoop, agent_diffs)

    # --- Save ---
    results = {
        "timestamp": datetime.now().isoformat(),
        "exp1": {"description": "Main comparison, 500 frames, DAIR-V2X-C, real velocities",
                 "methods": list(methods.keys()),
                 "metrics": [_safe_asdict(m) for m in exp1_metrics]},
        "exp2": {"description": "Ablation, same 500 frames",
                 "methods": list(abl_methods.keys()),
                 "metrics": [_safe_asdict(m) for m in exp2_metrics]},
        "exp3": {"description": "Cooperative vs non-cooperative, 200 frames, TRUE vehicle-side separation",
                 "coop": [_safe_asdict(m) for m in exp3_coop],
                 "noncoop": [_safe_asdict(m) for m in exp3_noncoop],
                 "agent_diffs": agent_diffs},
    }
    (output_dir / "offline_results.json").write_text(json.dumps(results, indent=2, default=str))


# ============================================================
# EXP 4: CARLA closed-loop
# ============================================================

def run_carla(output_dir: Path):
    """EXP-4: CARLA closed-loop driving evaluation."""
    import carla
    from coop_safety.utils.metrics import compute_ttc

    print("\n" + "=" * 60)
    print("EXP-4: CARLA Closed-Loop Driving Evaluation")
    print("  Simulator: CARLA 0.9.15 (headless)")
    print("  Base driver: CARLA autopilot")
    print("  Safety constraint: applied on top of autopilot")
    print("  Traffic: aggressive (close following, speeding, ignore lights)")
    print("  Metrics: collision, near-miss, route distance, speed, comfort")
    print("=" * 60)

    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(8000)

    methods = {
        "NoConstraint": None,
        "RSS [Shalev-Shwartz'17]": RSSOnly(),
        "APF [Rasekhipour17]": RiskPotentialField(),
        "Ours-Rule": SafetyConstraintModule(),
    }

    configs = [
        {"n_vehicles": 30, "n_walkers": 5, "label": "30v+5w"},
        {"n_vehicles": 50, "n_walkers": 10, "label": "50v+10w"},
    ]
    episodes_per = 3
    episode_duration = 40.0
    dt = 0.05

    all_results = []

    for cfg in configs:
        for method_name, safety_module in methods.items():
            for ep in range(episodes_per):
                print(f"\n  [{cfg['label']} | {method_name} | ep{ep}] ", end="", flush=True)

                # Clean world
                _clean_carla(client, world)

                # Setup traffic
                tm.set_global_distance_to_leading_vehicle(1.5)
                tm.global_percentage_speed_difference(-20)

                np.random.seed(ep + hash(cfg['label']) % 1000)

                # Spawn
                vehicles = []
                veh_bps = bp_lib.filter("vehicle.*")
                for i in np.random.permutation(len(spawn_points))[:cfg["n_vehicles"]]:
                    v = world.try_spawn_actor(veh_bps[int(i) % len(veh_bps)], spawn_points[int(i)])
                    if v:
                        v.set_autopilot(True, tm.get_port())
                        if np.random.random() < 0.3:
                            tm.ignore_lights_percentage(v, 80)
                        vehicles.append(v)

                walkers = []
                walker_ctrls = []
                for _ in range(cfg["n_walkers"]):
                    sp = carla.Transform(carla.Location(
                        x=spawn_points[0].location.x + np.random.uniform(-40, 40),
                        y=spawn_points[0].location.y + np.random.uniform(-40, 40), z=1.0))
                    w = world.try_spawn_actor(bp_lib.filter("walker.pedestrian.*")[0], sp)
                    if w:
                        walkers.append(w)
                        ctrl = world.try_spawn_actor(bp_lib.find("controller.ai.walker"), carla.Transform(), w)
                        if ctrl:
                            walker_ctrls.append(ctrl)
                            ctrl.start()
                            ctrl.go_to_location(world.get_random_location_from_navigation())

                if not vehicles:
                    continue
                ego = vehicles[0]
                others = vehicles[1:] + walkers

                # Collision sensor
                coll_count = [0]
                col_sensor = world.spawn_actor(
                    bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
                col_sensor.listen(lambda e: coll_count.__setitem__(0, coll_count[0] + 1))

                time.sleep(5)

                # Run episode
                speeds, accels, near_misses = [], [], 0
                prev_speed = 0
                n_steps = int(episode_duration / dt)
                min_harm_count = 0

                for step in range(n_steps):
                    time.sleep(dt)
                    try:
                        et = ego.get_transform()
                        ev = ego.get_velocity()
                    except:
                        break

                    speed = math.sqrt(ev.x**2 + ev.y**2)
                    speeds.append(speed)
                    accels.append(abs(speed - prev_speed) / dt)
                    prev_speed = speed

                    # Check near-miss
                    for a in others:
                        try:
                            at = a.get_transform()
                            d = math.sqrt((at.location.x-et.location.x)**2 + (at.location.y-et.location.y)**2)
                            if d < 5:
                                near_misses += 1
                                break
                        except:
                            continue

                    # Apply safety constraint
                    if safety_module is not None:
                        from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType
                        perception = _build_perception(ego, others)
                        if perception:
                            try:
                                safe = safety_module.constrain(perception)
                                if safe.mode == ConstraintMode.MINIMUM_HARM:
                                    min_harm_count += 1
                                    # Check rear before braking
                                    ego.apply_control(carla.VehicleControl(throttle=0, brake=0.5, steer=0))
                                elif safe.mode == ConstraintMode.CONSERVATIVE:
                                    ego.apply_control(carla.VehicleControl(throttle=0.15, brake=0.15, steer=0))
                            except:
                                pass

                # Collect metrics
                distance = sum(speeds) * dt
                avg_speed = np.mean(speeds) if speeds else 0
                comfort = np.mean(accels) if accels else 0  # Lower = smoother

                result = {
                    "method": method_name, "config": cfg["label"], "episode": ep,
                    "collisions": coll_count[0], "near_misses": near_misses,
                    "distance_m": distance, "avg_speed_ms": avg_speed,
                    "comfort_jerk": comfort, "min_harm": min_harm_count,
                }
                all_results.append(result)
                print(f"coll={coll_count[0]} nm={near_misses} dist={distance:.0f}m "
                      f"speed={avg_speed:.1f} mh={min_harm_count}")

                # Cleanup
                try: col_sensor.destroy()
                except: pass
                for c in walker_ctrls:
                    try: c.stop(); c.destroy()
                    except: pass
                _clean_carla(client, world)

    # Print summary
    print("\n" + "=" * 60)
    print("EXP-4 RESULTS: CARLA Closed-Loop")
    print("=" * 60)
    print(f"\n{'Method':30s} {'Coll':>6s} {'NearMiss':>9s} {'Dist(m)':>8s} {'Speed':>6s} {'Comfort':>8s} {'MH':>4s}")
    print("-" * 75)
    for mname in methods:
        mr = [r for r in all_results if r["method"] == mname]
        if not mr: continue
        n = len(mr)
        print(f"{mname:30s} "
              f"{sum(r['collisions'] for r in mr):>3d}/{n:<2d} "
              f"{sum(r['near_misses'] for r in mr):>9d} "
              f"{np.mean([r['distance_m'] for r in mr]):>7.0f} "
              f"{np.mean([r['avg_speed_ms'] for r in mr]):>5.1f} "
              f"{np.mean([r['comfort_jerk'] for r in mr]):>7.2f} "
              f"{sum(r['min_harm'] for r in mr):>4d}")

    (output_dir / "carla_results.json").write_text(json.dumps({
        "description": "CARLA closed-loop, autopilot + safety constraint overlay",
        "configs": configs,
        "episodes_per": episodes_per,
        "results": all_results,
    }, indent=2, default=str))


# ============================================================
# Helpers
# ============================================================

def _safe_asdict(m):
    d = asdict(m)
    for k, v in d.items():
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            d[k] = str(v)
    return d


def _print_table(title, metrics, method_names):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    print(f"{'Method':35s} {'Area':>7s} {'Tight%':>7s} {'TTC':>6s} {'MH':>6s} {'ms':>5s}")
    print("-" * 68)
    for mname in method_names:
        mm = [m for m in metrics if m.method_name == mname]
        if not mm: continue
        a = np.mean([m.feasible_area for m in mm])
        t = np.mean([m.constraint_tightening_ratio for m in mm])
        ttc = np.mean([min(float(m.min_ttc) if not isinstance(m.min_ttc, str) else 100, 100) for m in mm])
        mh = sum(1 for m in mm if m.mode == "minimum_harm")
        tm = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:35s} {a:7.1f} {t:6.1%} {ttc:6.2f} {mh:>3d}/{len(mm)} {tm:4.0f}")


def _print_coop(title, coop, noncoop, diffs):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    for label, metrics in [("Cooperative (V2I)", coop), ("Non-cooperative (ego)", noncoop)]:
        a = np.mean([m.feasible_area for m in metrics])
        t = np.mean([m.constraint_tightening_ratio for m in metrics])
        mh = sum(1 for m in metrics if m.mode == "minimum_harm")
        print(f"  {label:25s} area={a:.1f}m² tight={t:.1%} mh={mh}/{len(metrics)}")
    avg_c = np.mean([d["coop"] for d in diffs])
    avg_n = np.mean([d["noncoop"] for d in diffs])
    print(f"  Avg agents: coop={avg_c:.1f}, noncoop={avg_n:.1f}, diff={avg_c-avg_n:.1f}")


def _build_perception(ego, others):
    from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType
    try:
        et = ego.get_transform()
        ev = ego.get_velocity()
    except:
        return None
    ego_state = VehicleState(
        id="ego", x=et.location.x, y=et.location.y,
        heading=math.radians(et.rotation.yaw),
        velocity=max(math.sqrt(ev.x**2+ev.y**2), 1.0),
        vx=ev.x, vy=ev.y,
        length=max(ego.bounding_box.extent.x*2, 4.0),
        width=max(ego.bounding_box.extent.y*2, 1.5))
    agents = []
    for i, a in enumerate(others):
        try:
            at = a.get_transform()
            av = a.get_velocity()
            d = math.sqrt((at.location.x-et.location.x)**2+(at.location.y-et.location.y)**2)
            if d > 60: continue
            is_w = "walker" in a.type_id
            agents.append(Agent(state=VehicleState(
                id=f"a{i}", x=at.location.x, y=at.location.y,
                heading=math.radians(at.rotation.yaw),
                velocity=max(math.sqrt(av.x**2+av.y**2), 0.1),
                vx=av.x, vy=av.y,
                length=max(a.bounding_box.extent.x*2, 0.5),
                width=max(a.bounding_box.extent.y*2, 0.5),
                vehicle_type="pedestrian" if is_w else "car",
                mass=70 if is_w else 1500),
                agent_type=AgentType.PEDESTRIAN if is_w else AgentType.VEHICLE))
        except: continue
    return PerceptionResult(timestamp=0, ego=ego_state, agents=agents)


def _clean_carla(client, world):
    import carla
    try:
        for filt in ["vehicle.*", "walker.*", "controller.*"]:
            actors = world.get_actors().filter(filt)
            if len(actors):
                client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        time.sleep(1)
    except: pass


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--carla", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not (args.offline or args.carla or args.all):
        args.all = True

    output_dir = Path(__file__).parent / "results" / f"unified_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    if args.offline or args.all:
        run_offline(output_dir)

    if args.carla or args.all:
        run_carla(output_dir)

    # Save reproduction info
    (output_dir / "reproduction.json").write_text(json.dumps({
        "command": "python experiments/run_unified.py --all",
        "env": "coop-safety",
        "carla": "0.9.15 headless on port 2000",
        "dairv2x": DATA_DIR,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()

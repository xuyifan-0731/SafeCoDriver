"""SUMO Closed-Loop: Each method controls ego independently.

Each method runs its own SUMO simulation. The safety constraint's modified
waypoints are used to control ego's speed (via traci.vehicle.slowDown).

This allows measuring per-method collision rates and secondary collisions.

Secondary collision definition: each collision after the first counts as one
secondary collision (3 collisions = 2 secondary collisions).
"""
from __future__ import annotations

import os, sys, math, time, random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

sys.path.insert(0, str(Path(__file__).parent.parent))
from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType, ConstraintMode
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
EGO_RANGE = 40.0
COOP_RANGE = 40.0


def generate_network(scenario_type: str, output_dir: Path) -> Path:
    net_file = output_dir / f"{scenario_type}.net.xml"
    if net_file.exists():
        return net_file
    arms = 3 if scenario_type in ('t_junction', 'blind_pedestrian') else 4
    os.system(f"{NETGEN} --spider --spider.arm-number {arms} --spider.space-radius 60 "
              f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 1 "
              f"-o {net_file} 2>/dev/null")
    return net_file


def generate_route_file(net_file: Path, scenario_type: str, scenario_id: int,
                        n_vehicles: int = 8) -> Path:
    random.seed(42 + scenario_id * 7)
    rou_file = net_file.parent / f"{scenario_type}_{scenario_id}.rou.xml"

    net = sumolib.net.readNet(str(net_file))
    center_node = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3 and len(n.getOutgoing()) >= 3:
            center_node = n
            break

    valid_routes = []
    if center_node:
        for e_in in center_node.getIncoming():
            for e_out in e_in.getOutgoing():
                valid_routes.append((e_in.getID(), e_out.getID()))

    if not valid_routes:
        edges = [e.getID() for e in net.getEdges()]
        valid_routes = [(edges[0], edges[1])]

    vehicles = []
    ego_route = valid_routes[0]
    vehicles.append(f'    <vehicle id="ego" type="car" depart="0" departSpeed="12">'
                    f'\n        <route edges="{ego_route[0]} {ego_route[1]}"/>\n    </vehicle>')

    coop_route = valid_routes[min(2, len(valid_routes)-1)]
    vehicles.append(f'    <vehicle id="coop" type="car" depart="0" departSpeed="9">'
                    f'\n        <route edges="{coop_route[0]} {coop_route[1]}"/>\n    </vehicle>')

    for i in range(n_vehicles - 2):
        route = random.choice(valid_routes)
        depart = random.uniform(0.2, 3.0)
        speed = random.uniform(9, 14)
        vtype = "pedestrian" if (scenario_type == 'blind_pedestrian' and i < 2) else "car"
        dep_speed = "2.0" if vtype == "pedestrian" else f"{speed:.1f}"
        vehicles.append(
            f'    <vehicle id="veh{i}" type="{vtype}" depart="{depart:.1f}" departSpeed="{dep_speed}">'
            f'\n        <route edges="{route[0]} {route[1]}"/>\n    </vehicle>')

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" length="4.5" width="1.8" maxSpeed="16.67" accel="3.0" decel="6.0"
           sigma="0.8" minGap="0.3" tau="0.3" speedDev="0.2"/>
    <vType id="pedestrian" length="0.6" width="0.6" maxSpeed="2.0" accel="1.0" decel="2.0"
           sigma="0.9" minGap="0.1" tau="0.2"/>
{chr(10).join(vehicles)}
</routes>"""
    with open(rou_file, 'w') as f:
        f.write(content)
    return rou_file


def build_perception_from_sumo():
    """Build ego-centric perception from current SUMO state."""
    veh_ids = traci.vehicle.getIDList()
    if "ego" not in veh_ids:
        return None

    ex, ey = traci.vehicle.getPosition("ego")
    e_speed = traci.vehicle.getSpeed("ego")
    e_angle = traci.vehicle.getAngle("ego")
    e_heading = math.radians(90 - e_angle)
    ego_vx = e_speed * math.cos(e_heading)
    ego_vy = e_speed * math.sin(e_heading)

    coop_pos = None
    if "coop" in veh_ids:
        cx, cy = traci.vehicle.getPosition("coop")
        coop_pos = (cx, cy)

    agents = []
    for vid in veh_ids:
        if vid in ("ego", "coop"):
            continue
        try:
            vx_pos, vy_pos = traci.vehicle.getPosition(vid)
            v_speed = traci.vehicle.getSpeed(vid)
            v_angle = traci.vehicle.getAngle(vid)
            v_heading = math.radians(90 - v_angle)
            v_length = traci.vehicle.getLength(vid)
            v_width = traci.vehicle.getWidth(vid)
        except:
            continue

        dx = vx_pos - ex
        dy = vy_pos - ey
        dist_ego = math.sqrt(dx**2 + dy**2)
        dist_coop = 999
        if coop_pos:
            dist_coop = math.sqrt((vx_pos-coop_pos[0])**2 + (vy_pos-coop_pos[1])**2)

        visible_ego = dist_ego <= EGO_RANGE
        visible_coop = dist_coop <= COOP_RANGE
        if not visible_ego and not visible_coop:
            continue

        cos_h = math.cos(-e_heading)
        sin_h = math.sin(-e_heading)
        rel_x = dx * cos_h - dy * sin_h
        rel_y = dx * sin_h + dy * cos_h
        v_vx = v_speed * math.cos(v_heading)
        v_vy = v_speed * math.sin(v_heading)
        rel_vx = (v_vx - ego_vx) * cos_h - (v_vy - ego_vy) * sin_h
        rel_vy = (v_vx - ego_vx) * sin_h + (v_vy - ego_vy) * cos_h

        state = VehicleState(id=vid, x=rel_x, y=rel_y, heading=v_heading-e_heading,
                             velocity=v_speed, vx=rel_vx, vy=rel_vy,
                             length=v_length, width=v_width,
                             vehicle_type='pedestrian' if v_length < 1 else 'car')
        agents.append(Agent(state=state,
                           agent_type=AgentType.PEDESTRIAN if v_length < 1 else AgentType.VEHICLE,
                           is_visible=visible_ego, confidence=1.0 if visible_ego else 0.7))

    return PerceptionResult(
        timestamp=traci.simulation.getTime(),
        ego=VehicleState(id="ego", x=0, y=0, heading=0, velocity=e_speed,
                         vx=e_speed, vy=0, length=4.5, width=1.8),
        agents=agents)


def apply_safety_to_ego(method, perception, base_wp):
    """Apply safety constraint and return target speed for ego.

    If method detects danger or modifies waypoints significantly,
    slow down ego. Otherwise, maintain speed.
    """
    if method is None:
        return None  # NoConstraint: don't intervene

    try:
        if hasattr(method, 'constrain_waypoints'):
            mod_wp, stats = method.constrain_waypoints(base_wp, perception)
            # If waypoints were modified, compute target speed from modified trajectory
            if stats.get('n_collisions_detected', 0) > 0 or stats.get('n_geometric_threats', 0) > 0:
                # Slow down: target speed = distance to first modified wp / dt
                dist = math.sqrt(mod_wp[0, 0]**2 + mod_wp[0, 1]**2)
                target_speed = max(dist / 0.5, 0.0)  # Don't go negative
                return min(target_speed, perception.ego.velocity)  # Don't speed up
            return None
        else:
            safe = method.constrain(perception)
            if safe.mode == ConstraintMode.MINIMUM_HARM:
                return 0.0  # Emergency stop
            elif safe.mode == ConstraintMode.CONSERVATIVE:
                return max(perception.ego.velocity * 0.5, 0.0)
            return None
    except:
        return None


@dataclass
class MethodScenarioResult:
    n_frames: int = 0
    ego_collisions: int = 0  # total ego collision events
    secondary_collisions: int = 0  # collisions after first (N collisions → N-1 secondary)
    collision_severities: list = field(default_factory=list)
    first_warning_frame: int = -1
    n_warning_frames: int = 0
    wp_collisions: int = 0
    wp_total: int = 0


def run_scenario_for_method(net_file, rou_file, method, method_name, max_steps=300):
    """Run one scenario with one method controlling ego."""
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true',
                '--seed', '42']

    result = MethodScenarioResult()

    try:
        traci.start(sumo_cmd)
        ego_collision_count = 0

        for step in range(max_steps):
            traci.simulationStep()
            veh_ids = traci.vehicle.getIDList()

            if "ego" not in veh_ids:
                if step > 5:
                    break
                continue

            perception = build_perception_from_sumo()
            if perception is None:
                continue

            result.n_frames += 1

            # Generate base waypoints
            e_speed = perception.ego.velocity
            base_wp = np.zeros((10, 2))
            for t in range(10):
                base_wp[t, 0] = max(e_speed, 1.0) * (t + 1) * 0.5

            # Apply safety constraint → get target speed
            target_speed = apply_safety_to_ego(method, perception, base_wp)

            # Apply speed control to ego in SUMO
            if target_speed is not None:
                try:
                    traci.vehicle.slowDown("ego", max(target_speed, 0), 1.0)
                except:
                    pass

                if result.first_warning_frame < 0:
                    result.first_warning_frame = step
                result.n_warning_frames += 1

            # Check waypoint collisions (on modified or base waypoints)
            if method and hasattr(method, 'constrain_waypoints'):
                try:
                    mod_wp, _ = method.constrain_waypoints(base_wp, perception)
                except:
                    mod_wp = base_wp
            else:
                mod_wp = base_wp

            for t in range(len(mod_wp)):
                dt = (t + 1) * 0.5
                for a in perception.agents:
                    ax = a.state.x + a.state.vx * dt
                    ay = a.state.y + a.state.vy * dt
                    if math.sqrt((mod_wp[t,0]-ax)**2 + (mod_wp[t,1]-ay)**2) < 2.0:
                        result.wp_collisions += 1
                        break
            result.wp_total += 10

            # Check collisions
            try:
                for col in traci.simulation.getCollisions():
                    if col.collider == "ego" or col.victim == "ego":
                        ego_collision_count += 1
                        try:
                            s1 = traci.vehicle.getSpeed(col.collider) if col.collider in veh_ids else 0
                            s2 = traci.vehicle.getSpeed(col.victim) if col.victim in veh_ids else 0
                            result.collision_severities.append(abs(s1 + s2))
                        except:
                            result.collision_severities.append(10.0)
            except:
                pass

        traci.close()
    except Exception as e:
        try: traci.close()
        except: pass

    result.ego_collisions = ego_collision_count
    # Secondary collisions: N collisions → N-1 secondary (first one is "primary")
    result.secondary_collisions = max(ego_collision_count - 1, 0)
    return result


def main():
    print("="*70)
    print("  SUMO Closed-Loop: Per-Method Secondary Collision Analysis")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "APF": RiskPotentialField(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(detector_model=v1,
                                               base_margin_visible=2.5,
                                               base_margin_invisible=4.0,
                                               detection_threshold=0.3),
    }

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_configs = [
        ("t_junction", 15),
        ("crossroads", 15),
        ("blind_pedestrian", 10),
    ]

    # Results: method_name → scenario_type → list of MethodScenarioResult
    all_results = {m: {s: [] for s, _, in scenario_configs} for m in methods}

    t0 = time.time()

    for stype, n_sc in scenario_configs:
        print(f"\n  --- {stype} ({n_sc} scenarios × {len(methods)} methods) ---")
        net_file = generate_network(stype, output_dir)

        for sid in range(n_sc):
            rou_file = generate_route_file(net_file, stype, sid,
                                           n_vehicles=random.randint(6, 10))

            for mname, method in methods.items():
                result = run_scenario_for_method(net_file, rou_file, method, mname)
                all_results[mname][stype].append(result)

            if sid % 5 == 0:
                # Print progress for Ours-Hybrid
                r = all_results["Ours-Hybrid"][stype][-1]
                print(f"    [{sid}/{n_sc}] Ours: coll={r.ego_collisions} secondary={r.secondary_collisions}")

    # ==========================================================================
    # Print results
    # ==========================================================================
    print("\n" + "="*80)
    print("  RESULTS: Per-Method Collision & Secondary Collision Analysis")
    print("="*80)

    for stype, n_sc in scenario_configs:
        print(f"\n  [{stype}] {n_sc} scenarios (closed-loop, each method controls ego)")
        print(f"  {'Method':20s} {'EgoColl':>8s} {'Secondary':>10s} {'SecRate':>8s} "
              f"{'AvgSeverity':>12s} {'WPColl%':>8s} {'DetRate':>8s}")
        print("  " + "-"*75)

        for mname in methods:
            results = all_results[mname][stype]
            total_coll = sum(r.ego_collisions for r in results)
            total_sec = sum(r.secondary_collisions for r in results)
            sec_rate = total_sec / max(total_coll, 1)
            severities = [s for r in results for s in r.collision_severities]
            avg_sev = np.mean(severities) if severities else 0

            # Detection: scenarios with collisions where method warned before
            n_coll_scenarios = sum(1 for r in results if r.ego_collisions > 0)
            n_detected = sum(1 for r in results if r.ego_collisions > 0 and r.first_warning_frame >= 0)
            det_rate = n_detected / max(n_coll_scenarios, 1)

            total_wp = sum(r.wp_total for r in results)
            total_wpc = sum(r.wp_collisions for r in results)
            wp_rate = total_wpc / max(total_wp, 1)

            print(f"  {mname:20s} {total_coll:8d} {total_sec:10d} {sec_rate:7.0%} "
                  f"{avg_sev:11.1f} {wp_rate:7.1%} {det_rate:7.1%}")

    # Overall
    print("\n" + "="*80)
    print("  OVERALL (all scenarios)")
    print("="*80)
    print(f"  {'Method':20s} {'EgoColl':>8s} {'Secondary':>10s} {'SecRate':>8s} "
          f"{'AvgSeverity':>12s} {'WPColl%':>8s}")
    print("  " + "-"*70)

    for mname in methods:
        total_coll = 0; total_sec = 0; severities = []; tw = 0; twc = 0
        for stype, _ in scenario_configs:
            for r in all_results[mname][stype]:
                total_coll += r.ego_collisions
                total_sec += r.secondary_collisions
                severities.extend(r.collision_severities)
                tw += r.wp_total
                twc += r.wp_collisions
        sec_rate = total_sec / max(total_coll, 1)
        avg_sev = np.mean(severities) if severities else 0
        wp_rate = twc / max(tw, 1)
        print(f"  {mname:20s} {total_coll:8d} {total_sec:10d} {sec_rate:7.0%} "
              f"{avg_sev:11.1f} {wp_rate:7.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

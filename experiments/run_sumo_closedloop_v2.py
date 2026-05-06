"""SUMO Closed-Loop v2: ALL vehicles use safety constraint.

Key change from v1: Every vehicle (not just ego) applies the same safety
constraint method. This prevents the "ego brakes → rear-end collision" problem
because following vehicles also brake when they detect danger.

Cooperative perception: only ego + coop share perception.
Other vehicles use their own local perception (40m range, no cooperation).
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
OTHER_VEH_RANGE = 40.0  # non-coop vehicles see 40m (no cooperation)


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
    rou_file = net_file.parent / f"{scenario_type}_v2_{scenario_id}.rou.xml"

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
           sigma="0.8" minGap="0.5" tau="0.5" speedDev="0.2"/>
    <vType id="pedestrian" length="0.6" width="0.6" maxSpeed="2.0" accel="1.0" decel="2.0"
           sigma="0.9" minGap="0.1" tau="0.2"/>
{chr(10).join(vehicles)}
</routes>"""
    with open(rou_file, 'w') as f:
        f.write(content)
    return rou_file


def get_veh_state(vid):
    """Get global state of a vehicle."""
    try:
        x, y = traci.vehicle.getPosition(vid)
        speed = traci.vehicle.getSpeed(vid)
        angle = traci.vehicle.getAngle(vid)
        heading = math.radians(90 - angle)
        length = traci.vehicle.getLength(vid)
        width = traci.vehicle.getWidth(vid)
        return x, y, speed, heading, length, width
    except:
        return None


def build_perception_for_vehicle(target_vid: str, veh_ids: list,
                                  coop_vid: str = None) -> PerceptionResult:
    """Build ego-centric perception for any vehicle.

    target_vid sees within its own 40m range.
    If coop_vid is provided, also includes agents seen by coop within 40m.
    """
    target_state = get_veh_state(target_vid)
    if target_state is None:
        return None
    tx, ty, t_speed, t_heading, t_length, t_width = target_state
    t_vx = t_speed * math.cos(t_heading)
    t_vy = t_speed * math.sin(t_heading)

    coop_pos = None
    if coop_vid and coop_vid in veh_ids and coop_vid != target_vid:
        cs = get_veh_state(coop_vid)
        if cs:
            coop_pos = (cs[0], cs[1])

    agents = []
    for vid in veh_ids:
        if vid == target_vid or vid == coop_vid:
            continue
        vs = get_veh_state(vid)
        if vs is None:
            continue
        vx_pos, vy_pos, v_speed, v_heading, v_length, v_width = vs

        dx = vx_pos - tx
        dy = vy_pos - ty
        dist_target = math.sqrt(dx**2 + dy**2)

        visible_target = dist_target <= OTHER_VEH_RANGE
        visible_coop = False
        if coop_pos:
            dist_coop = math.sqrt((vx_pos-coop_pos[0])**2 + (vy_pos-coop_pos[1])**2)
            visible_coop = dist_coop <= COOP_RANGE

        if not visible_target and not visible_coop:
            continue

        # Convert to target-centric frame
        cos_h = math.cos(-t_heading)
        sin_h = math.sin(-t_heading)
        rel_x = dx * cos_h - dy * sin_h
        rel_y = dx * sin_h + dy * cos_h
        v_vx = v_speed * math.cos(v_heading)
        v_vy = v_speed * math.sin(v_heading)
        rel_vx = (v_vx - t_vx) * cos_h - (v_vy - t_vy) * sin_h
        rel_vy = (v_vx - t_vx) * sin_h + (v_vy - t_vy) * cos_h

        state = VehicleState(id=vid, x=rel_x, y=rel_y, heading=v_heading - t_heading,
                             velocity=v_speed, vx=rel_vx, vy=rel_vy,
                             length=v_length, width=v_width,
                             vehicle_type='pedestrian' if v_length < 1 else 'car')
        agents.append(Agent(state=state,
                           agent_type=AgentType.PEDESTRIAN if v_length < 1 else AgentType.VEHICLE,
                           is_visible=visible_target,
                           confidence=1.0 if visible_target else 0.7))

    return PerceptionResult(
        timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid, x=0, y=0, heading=0, velocity=t_speed,
                         vx=t_speed, vy=0, length=t_length, width=t_width),
        agents=agents)


def compute_target_speed(method, perception, current_speed):
    """Apply safety constraint and compute target speed."""
    if method is None or perception is None:
        return None

    base_wp = np.zeros((10, 2))
    for t in range(10):
        base_wp[t, 0] = max(current_speed, 1.0) * (t + 1) * 0.5

    try:
        if hasattr(method, 'constrain_waypoints'):
            mod_wp, stats = method.constrain_waypoints(base_wp, perception)
            if stats.get('n_collisions_detected', 0) > 0 or stats.get('n_geometric_threats', 0) > 0:
                dist = math.sqrt(mod_wp[0, 0]**2 + mod_wp[0, 1]**2)
                target = max(dist / 0.5, 0.0)
                return min(target, current_speed)
        else:
            safe = method.constrain(perception)
            if safe.mode == ConstraintMode.MINIMUM_HARM:
                return max(current_speed * 0.3, 0.0)
            elif safe.mode == ConstraintMode.CONSERVATIVE:
                return max(current_speed * 0.6, 0.0)
    except:
        pass
    return None


@dataclass
class MethodScenarioResult:
    n_frames: int = 0
    ego_collisions: int = 0
    secondary_collisions: int = 0
    collision_severities: list = field(default_factory=list)
    first_warning_frame: int = -1
    n_warning_frames: int = 0
    wp_collisions: int = 0
    wp_total: int = 0


def run_scenario_for_method(net_file, rou_file, method, method_name, max_steps=300):
    """Run scenario: ALL vehicles use the same safety constraint method."""
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
            veh_ids = list(traci.vehicle.getIDList())

            if "ego" not in veh_ids:
                if step > 5:
                    break
                continue

            result.n_frames += 1

            # Apply safety constraint to ALL vehicles (not just ego)
            for vid in veh_ids:
                if method is None:
                    continue  # NoConstraint: nobody intervenes

                # Ego and coop share perception; others use local-only
                if vid == "ego":
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid="coop")
                elif vid == "coop":
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid="ego")
                else:
                    # Other vehicles: local perception only (no cooperation)
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid=None)

                if perception is None:
                    continue

                current_speed = perception.ego.velocity
                target_speed = compute_target_speed(method, perception, current_speed)

                if target_speed is not None:
                    try:
                        traci.vehicle.slowDown(vid, max(target_speed, 0), 1.0)
                    except:
                        pass

                    # Track warnings for ego only
                    if vid == "ego":
                        if result.first_warning_frame < 0:
                            result.first_warning_frame = step
                        result.n_warning_frames += 1

            # Check ego waypoint collisions
            ego_perc = build_perception_for_vehicle("ego", veh_ids, coop_vid="coop")
            if ego_perc:
                base_wp = np.zeros((10, 2))
                e_speed = ego_perc.ego.velocity
                for t in range(10):
                    base_wp[t, 0] = max(e_speed, 1.0) * (t + 1) * 0.5

                if method and hasattr(method, 'constrain_waypoints'):
                    try:
                        mod_wp, _ = method.constrain_waypoints(base_wp, ego_perc)
                    except:
                        mod_wp = base_wp
                else:
                    mod_wp = base_wp

                for t in range(len(mod_wp)):
                    dt = (t + 1) * 0.5
                    for a in ego_perc.agents:
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
    result.secondary_collisions = max(ego_collision_count - 1, 0)
    return result


def main():
    print("="*70)
    print("  SUMO Closed-Loop v2: ALL Vehicles Use Safety Constraint")
    print("="*70)
    print("  Design: every vehicle applies the same method (not just ego)")
    print("  Cooperation: only ego + coop share perception")

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

    all_results = {m: {s: [] for s, _ in scenario_configs} for m in methods}
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
                r = all_results["Ours-Hybrid"][stype][-1]
                nc = all_results["NoConstraint"][stype][-1]
                print(f"    [{sid}/{n_sc}] NoCon: coll={nc.ego_collisions} "
                      f"Ours: coll={r.ego_collisions} sec={r.secondary_collisions}")

    # Print results
    print("\n" + "="*80)
    print("  RESULTS: All Vehicles Use Safety Constraint (Closed-Loop)")
    print("="*80)

    for stype, n_sc in scenario_configs:
        print(f"\n  [{stype}] {n_sc} scenarios")
        print(f"  {'Method':20s} {'EgoColl':>8s} {'2ndColl':>8s} {'2ndRate':>8s} "
              f"{'Severity':>9s} {'WPColl%':>8s} {'DetRate':>8s}")
        print("  " + "-"*70)

        for mname in methods:
            results = all_results[mname][stype]
            tc = sum(r.ego_collisions for r in results)
            ts = sum(r.secondary_collisions for r in results)
            sr = ts / max(tc, 1)
            sevs = [s for r in results for s in r.collision_severities]
            avg_sev = np.mean(sevs) if sevs else 0
            ncs = sum(1 for r in results if r.ego_collisions > 0)
            nd = sum(1 for r in results if r.ego_collisions > 0 and r.first_warning_frame >= 0)
            dr = nd / max(ncs, 1)
            tw = sum(r.wp_total for r in results)
            twc = sum(r.wp_collisions for r in results)
            wr = twc / max(tw, 1)
            print(f"  {mname:20s} {tc:8d} {ts:8d} {sr:7.0%} "
                  f"{avg_sev:8.1f} {wr:7.1%} {dr:7.1%}")

    # Overall
    print("\n" + "="*80)
    print("  OVERALL (all scenarios)")
    print("="*80)
    print(f"  {'Method':20s} {'EgoColl':>8s} {'2ndColl':>8s} {'2ndRate':>8s} "
          f"{'Severity':>9s} {'WPColl%':>8s}")
    print("  " + "-"*65)

    for mname in methods:
        tc=0; ts=0; sevs=[]; tw=0; twc=0
        for stype, _ in scenario_configs:
            for r in all_results[mname][stype]:
                tc += r.ego_collisions; ts += r.secondary_collisions
                sevs.extend(r.collision_severities)
                tw += r.wp_total; twc += r.wp_collisions
        sr = ts / max(tc, 1)
        avg_sev = np.mean(sevs) if sevs else 0
        wr = twc / max(tw, 1)
        print(f"  {mname:20s} {tc:8d} {ts:8d} {sr:7.0%} "
              f"{avg_sev:8.1f} {wr:7.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

"""SUMO Extreme Scenario Simulation for SafeCoDriver.

Uses SUMO's built-in networks (spider topology) with aggressive vehicle
behavior to create collision-prone scenarios.

Scenarios:
  A. T-junction (3-arm spider): vehicles converge from 3 directions
  B. Crossroads (4-arm spider): vehicles converge from 4 directions
  C. Blind-spot emergence: slow pedestrian-type vehicles appear suddenly

Perception model:
  - Ego: detects agents within 40m (is_visible=True)
  - Coop vehicle: detects agents in its own 40m
  - Agents seen only by coop: is_visible=False
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
    """Generate SUMO network using netgenerate."""
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
    """Generate route file with aggressive departures using valid connections."""
    random.seed(42 + scenario_id * 7)
    rou_file = net_file.parent / f"{scenario_type}_{scenario_id}.rou.xml"

    net = sumolib.net.readNet(str(net_file))

    # Build valid routes: inward_edge -> outward_edge (only if connected)
    center_node = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3 and len(n.getOutgoing()) >= 3:
            center_node = n
            break

    valid_routes = []  # list of (in_edge_id, out_edge_id)
    if center_node:
        for e_in in center_node.getIncoming():
            outgoing = e_in.getOutgoing()
            for e_out in outgoing:
                valid_routes.append((e_in.getID(), e_out.getID()))

    if not valid_routes:
        # Fallback: just use first few edges
        edges = [e.getID() for e in net.getEdges()]
        valid_routes = [(edges[0], edges[1])]

    vehicles = []

    # Ego: first valid route
    ego_route = valid_routes[0]
    vehicles.append(f'    <vehicle id="ego" type="car" depart="0" departSpeed="12">'
                    f'\n        <route edges="{ego_route[0]} {ego_route[1]}"/>\n    </vehicle>')

    # Coop: different route (conflicting direction)
    coop_route = valid_routes[min(2, len(valid_routes)-1)]
    vehicles.append(f'    <vehicle id="coop" type="car" depart="0" departSpeed="9">'
                    f'\n        <route edges="{coop_route[0]} {coop_route[1]}"/>\n    </vehicle>')

    # Other vehicles: random valid routes, aggressive timing
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


@dataclass
class ScenarioResult:
    scenario_type: str
    scenario_id: int
    n_frames: int = 0
    total_collisions: int = 0
    ego_collisions: int = 0
    secondary_collisions: int = 0
    collision_severities: list = field(default_factory=list)
    method_results: dict = field(default_factory=dict)


def run_one_scenario(net_file, rou_file, scenario_type, scenario_id, methods, max_steps=300):
    """Run one SUMO scenario."""
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true']

    result = ScenarioResult(scenario_type=scenario_type, scenario_id=scenario_id)
    for name in methods:
        result.method_results[name] = {
            'first_warning': -1, 'n_warnings': 0,
            'wp_collisions': 0, 'wp_total': 0, 'modifications': 0}

    try:
        traci.start(sumo_cmd)
        ego_first_coll_step = -1

        for step in range(max_steps):
            traci.simulationStep()
            veh_ids = traci.vehicle.getIDList()

            if "ego" not in veh_ids:
                if step > 5:
                    break
                continue

            # Get ego state
            ex, ey = traci.vehicle.getPosition("ego")
            e_speed = traci.vehicle.getSpeed("ego")
            e_angle = traci.vehicle.getAngle("ego")
            e_heading = math.radians(90 - e_angle)

            # Get coop state
            coop_pos = None
            if "coop" in veh_ids:
                cx, cy = traci.vehicle.getPosition("coop")
                coop_pos = (cx, cy)

            # Build perception
            agents = []
            ego_vx = e_speed * math.cos(e_heading)
            ego_vy = e_speed * math.sin(e_heading)

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

                # Distance to ego
                dx = vx_pos - ex
                dy = vy_pos - ey
                dist_ego = math.sqrt(dx**2 + dy**2)

                # Distance to coop
                dist_coop = 999
                if coop_pos:
                    dist_coop = math.sqrt((vx_pos-coop_pos[0])**2 + (vy_pos-coop_pos[1])**2)

                visible_ego = dist_ego <= EGO_RANGE
                visible_coop = dist_coop <= COOP_RANGE

                if not visible_ego and not visible_coop:
                    continue

                # Convert to ego frame
                cos_h = math.cos(-e_heading)
                sin_h = math.sin(-e_heading)
                rel_x = dx * cos_h - dy * sin_h
                rel_y = dx * sin_h + dy * cos_h

                v_vx = v_speed * math.cos(v_heading)
                v_vy = v_speed * math.sin(v_heading)
                rel_vx = (v_vx - ego_vx) * cos_h - (v_vy - ego_vy) * sin_h
                rel_vy = (v_vx - ego_vx) * sin_h + (v_vy - ego_vy) * cos_h

                state = VehicleState(
                    id=vid, x=rel_x, y=rel_y, heading=v_heading - e_heading,
                    velocity=v_speed, vx=rel_vx, vy=rel_vy,
                    length=v_length, width=v_width,
                    vehicle_type='pedestrian' if 'ped' in vid.lower() or v_length < 1 else 'car')

                agents.append(Agent(
                    state=state,
                    agent_type=AgentType.PEDESTRIAN if v_length < 1 else AgentType.VEHICLE,
                    is_visible=visible_ego,
                    confidence=1.0 if visible_ego else 0.7,
                    source="ego" if visible_ego else "coop"))

            perception = PerceptionResult(
                timestamp=step * 0.1,
                ego=VehicleState(id="ego", x=0, y=0, heading=0, velocity=e_speed,
                                 vx=e_speed, vy=0, length=4.5, width=1.8),
                agents=agents)

            result.n_frames += 1

            # Check collisions
            try:
                colls = traci.simulation.getCollisions()
                for col in colls:
                    result.total_collisions += 1
                    # Estimate severity
                    try:
                        s1 = traci.vehicle.getSpeed(col.collider) if col.collider in veh_ids else 0
                        s2 = traci.vehicle.getSpeed(col.victim) if col.victim in veh_ids else 0
                        severity = abs(s1 + s2)
                    except:
                        severity = 10.0
                    result.collision_severities.append(severity)

                    if col.collider == "ego" or col.victim == "ego":
                        result.ego_collisions += 1
                        if ego_first_coll_step < 0:
                            ego_first_coll_step = step
                        elif step > ego_first_coll_step + 5:
                            result.secondary_collisions += 1
            except:
                pass

            # Generate waypoints
            base_wp = np.zeros((10, 2))
            for t in range(10):
                base_wp[t, 0] = max(e_speed, 1.0) * (t + 1) * 0.5

            # Evaluate methods
            for name, method in methods.items():
                mr = result.method_results[name]

                if method is None:
                    mr['wp_total'] += 10
                    continue

                try:
                    if hasattr(method, 'constrain_waypoints'):
                        mod_wp, stats = method.constrain_waypoints(base_wp, perception)
                        was_mod = stats.get('n_collisions_detected', 0) > 0 or stats.get('modification_rate', 0) > 0
                    else:
                        safe = method.constrain(perception)
                        was_mod = safe.mode != ConstraintMode.NORMAL
                        mod_wp = base_wp
                except:
                    was_mod = False
                    mod_wp = base_wp

                if was_mod:
                    if mr['first_warning'] < 0:
                        mr['first_warning'] = step
                    mr['n_warnings'] += 1
                    mr['modifications'] += 1

                # Check waypoint collisions
                for t in range(len(mod_wp)):
                    dt = (t + 1) * 0.5
                    for a in perception.agents:
                        ax = a.state.x + a.state.vx * dt
                        ay = a.state.y + a.state.vy * dt
                        if math.sqrt((mod_wp[t,0]-ax)**2 + (mod_wp[t,1]-ay)**2) < 2.0:
                            mr['wp_collisions'] += 1
                            break
                mr['wp_total'] += 10

        traci.close()
    except Exception as e:
        try: traci.close()
        except: pass
        if "Connection closed" not in str(e):
            print(f"    Error: {e}")

    return result


def main():
    print("="*70)
    print("  SUMO Extreme Scenario Evaluation — SafeCoDriver")
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
        ("t_junction", 20),
        ("crossroads", 20),
        ("blind_pedestrian", 10),
    ]

    all_results = []
    t0 = time.time()

    for stype, n_sc in scenario_configs:
        print(f"\n  --- {stype} ({n_sc} scenarios) ---")
        net_file = generate_network(stype, output_dir)

        for sid in range(n_sc):
            rou_file = generate_route_file(net_file, stype, sid,
                                           n_vehicles=random.randint(6, 12))
            result = run_one_scenario(net_file, rou_file, stype, sid, methods)
            all_results.append(result)

            if sid % 5 == 0:
                print(f"    [{sid}/{n_sc}] frames={result.n_frames} "
                      f"ego_coll={result.ego_collisions} secondary={result.secondary_collisions}")

    # Print results
    print("\n" + "="*80)
    print("  RESULTS BY SCENARIO TYPE")
    print("="*80)

    for stype in ["t_junction", "crossroads", "blind_pedestrian"]:
        type_results = [r for r in all_results if r.scenario_type == stype]
        if not type_results:
            continue

        n = len(type_results)
        ego_coll = sum(r.ego_collisions for r in type_results)
        secondary = sum(r.secondary_collisions for r in type_results)
        severities = [s for r in type_results for s in r.collision_severities]

        print(f"\n  [{stype}] {n} scenarios, {sum(r.n_frames for r in type_results)} frames")
        print(f"    Ego collisions: {ego_coll}, Secondary: {secondary} "
              f"({secondary/max(ego_coll,1):.0%} chain rate)")
        if severities:
            print(f"    Severity: mean={np.mean(severities):.1f}, max={np.max(severities):.1f} m/s")

        n_coll_sc = sum(1 for r in type_results if r.ego_collisions > 0)
        n_safe_sc = n - n_coll_sc

        print(f"    Collision scenarios: {n_coll_sc}/{n}, Safe: {n_safe_sc}/{n}")
        print(f"\n    {'Method':20s} {'DetRate':>8s} {'EarlyW':>7s} {'FA':>6s} "
              f"{'WPColl%':>8s} {'ModRate':>8s}")
        print("    " + "-"*55)

        for mname in methods:
            nd=0; ncs=0; nn=0; nfa=0; tw=0; twc=0; tm=0; tf=0; ew=[]
            for r in type_results:
                mr = r.method_results[mname]
                tf += r.n_frames; tw += mr['wp_total']
                twc += mr['wp_collisions']; tm += mr['modifications']
                if r.ego_collisions > 0:
                    ncs += 1
                    if mr['first_warning'] >= 0:
                        nd += 1; ew.append(max(r.n_frames - mr['first_warning'], 0))
                else:
                    nn += 1
                    if mr['n_warnings'] > 0: nfa += 1

            dr = nd/max(ncs,1); ae = np.mean(ew) if ew else 0
            fr = nfa/max(nn,1); wr = twc/max(tw,1); mr2 = tm/max(tf,1)
            print(f"    {mname:20s} {dr:7.1%} {ae:6.1f} {fr:5.1%} {wr:7.1%} {mr2:7.1%}")

    # Overall
    print("\n" + "="*80)
    print("  OVERALL")
    print("="*80)
    total_ego = sum(r.ego_collisions for r in all_results)
    total_sec = sum(r.secondary_collisions for r in all_results)
    print(f"  {len(all_results)} scenarios, ego_collisions={total_ego}, "
          f"secondary={total_sec} (chain rate: {total_sec/max(total_ego,1):.0%})")

    print(f"\n  {'Method':20s} {'DetRate':>8s} {'EarlyW':>7s} {'FA':>6s} "
          f"{'WPColl%':>8s} {'ModRate':>8s}")
    print("  " + "-"*55)
    for mname in methods:
        nd=0;ncs=0;nn=0;nfa=0;tw=0;twc=0;tm=0;tf=0;ew=[]
        for r in all_results:
            mr=r.method_results[mname]; tf+=r.n_frames; tw+=mr['wp_total']
            twc+=mr['wp_collisions']; tm+=mr['modifications']
            if r.ego_collisions>0:
                ncs+=1
                if mr['first_warning']>=0: nd+=1; ew.append(max(r.n_frames-mr['first_warning'],0))
            else:
                nn+=1
                if mr['n_warnings']>0: nfa+=1
        dr=nd/max(ncs,1);ae=np.mean(ew) if ew else 0
        fr=nfa/max(nn,1);wr=twc/max(tw,1);mr2=tm/max(tf,1)
        print(f"  {mname:20s} {dr:7.1%} {ae:6.1f} {fr:5.1%} {wr:7.1%} {mr2:7.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

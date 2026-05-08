"""SUMO v4: Dual-lane + lateral avoidance + collision deduplication.

Key changes from v2:
1. Dual-lane network (2 lanes per direction) — enables real lane changes
2. Lateral avoidance: vehicles can change lanes to avoid frontal threats
3. Collision deduplication: same (collider, victim) pair only counted ONCE
4. All vehicles use CoDriving + safety constraint
5. Focus on reducing secondary collisions

Collision counting rules:
- A collision between pair (A, B) is counted only ONCE per scenario
- If A and B collide again later (same step or subsequent), it's ignored
- A NEW secondary collision requires a DIFFERENT pair involving ego
  e.g., (ego, veh1) then (ego, veh2) = 1 primary + 1 secondary
"""
from __future__ import annotations
import sys, os, math, time, random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

sys.path.insert(0, str(Path(__file__).parent.parent))
from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType, ConstraintMode
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from experiments.methods import RSSOnly
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
EGO_RANGE = 50.0
COOP_RANGE = 50.0
OTHER_RANGE = 40.0


# =================================================================
# Network & Route Generation (dual-lane)
# =================================================================

def get_or_create_network(stype: str, output_dir: Path) -> Path:
    """Get dual-lane network (already generated or create)."""
    net_file = output_dir / f"{stype}_2lane.net.xml"
    if net_file.exists():
        return net_file
    arms = 3 if stype == 't_junction' else 4
    os.system(f"{NETGEN} --spider --spider.arm-number {arms} --spider.space-radius 80 "
              f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 2 "
              f"-o {net_file} 2>/dev/null")
    return net_file


def generate_routes(net_file: Path, stype: str, sid: int, n_veh: int = 10) -> Path:
    """Generate routes with vehicles on both lanes."""
    random.seed(42 + sid * 13)
    rou_file = net_file.parent / f"{stype}_v4_{sid}.rou.xml"

    net = sumolib.net.readNet(str(net_file))
    center = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3:
            center = n; break

    routes = []
    if center:
        for ei in center.getIncoming():
            for eo in ei.getOutgoing():
                routes.append((ei.getID(), eo.getID()))
    if not routes:
        routes = [('E0', 'E1')]

    vehs = []
    # Ego on lane 0
    ego_route = routes[0]
    vehs.append(f'    <vehicle id="ego" type="car" depart="0" departSpeed="11" departLane="0">'
                f'\n        <route edges="{ego_route[0]} {ego_route[1]}"/>\n    </vehicle>')
    # Coop on different approach, lane 1
    coop_route = routes[min(3, len(routes)-1)]
    vehs.append(f'    <vehicle id="coop" type="car" depart="0" departSpeed="9" departLane="1">'
                f'\n        <route edges="{coop_route[0]} {coop_route[1]}"/>\n    </vehicle>')
    # Other vehicles: random routes, random lanes, mixed types
    for i in range(n_veh - 2):
        r = random.choice(routes)
        d = random.uniform(0.1, 2.5)
        sp = random.uniform(10, 18)
        lane = random.randint(0, 1)
        vtype = "aggressive" if random.random() < 0.4 else "car"
        vehs.append(f'    <vehicle id="veh{i}" type="{vtype}" depart="{d:.1f}" '
                    f'departSpeed="{sp:.1f}" departLane="{lane}">'
                    f'\n        <route edges="{r[0]} {r[1]}"/>\n    </vehicle>')

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="car" length="4.5" width="1.8" maxSpeed="19.44" accel="4.0" decel="6.0"
           sigma="0.9" minGap="0.3" tau="0.3" lcStrategic="1.0" lcCooperative="0.0"
           lcSpeedGain="2.0" lcKeepRight="0.0" speedDev="0.3" impatience="0.9"/>
    <vType id="aggressive" length="4.5" width="1.8" maxSpeed="22.22" accel="5.0" decel="7.0"
           sigma="1.0" minGap="0.2" tau="0.2" lcStrategic="0.5" lcCooperative="0.0"
           lcSpeedGain="3.0" speedDev="0.4" impatience="1.0"/>
{chr(10).join(vehs)}
</routes>"""
    open(rou_file, 'w').write(content)
    return rou_file


# =================================================================
# Perception & Control
# =================================================================

def get_veh_info(vid):
    try:
        x, y = traci.vehicle.getPosition(vid)
        sp = traci.vehicle.getSpeed(vid)
        ang = traci.vehicle.getAngle(vid)
        h = math.radians(90 - ang)
        l = traci.vehicle.getLength(vid)
        w = traci.vehicle.getWidth(vid)
        return (x, y, sp, h, l, w)
    except:
        return None


def build_perception(target_vid, veh_ids, coop_vid=None):
    """Build ego-centric perception for target vehicle."""
    ts = get_veh_info(target_vid)
    if ts is None:
        return None
    tx, ty, tsp, th, tl, tw = ts
    tvx, tvy = tsp * math.cos(th), tsp * math.sin(th)

    cp = None
    if coop_vid and coop_vid in veh_ids:
        ci = get_veh_info(coop_vid)
        if ci: cp = (ci[0], ci[1])

    agents = []
    for vid in veh_ids:
        if vid in (target_vid, coop_vid):
            continue
        vi = get_veh_info(vid)
        if vi is None:
            continue
        vx, vy, vs, vh, vl, vw = vi
        dx, dy = vx - tx, vy - ty
        de = math.sqrt(dx**2 + dy**2)

        # Visibility check
        vis_target = de <= (EGO_RANGE if target_vid in ("ego", "coop") else OTHER_RANGE)
        vis_coop = False
        if cp:
            vis_coop = math.sqrt((vx-cp[0])**2 + (vy-cp[1])**2) <= COOP_RANGE
        if not vis_target and not vis_coop:
            continue

        # Transform to target frame
        ch, sh = math.cos(-th), math.sin(-th)
        rx = dx*ch - dy*sh
        ry = dx*sh + dy*ch
        vvx, vvy = vs*math.cos(vh), vs*math.sin(vh)
        rvx = (vvx-tvx)*ch - (vvy-tvy)*sh
        rvy = (vvx-tvx)*sh + (vvy-tvy)*ch

        agents.append(Agent(
            state=VehicleState(id=vid, x=rx, y=ry, heading=vh-th,
                              velocity=vs, vx=rvx, vy=rvy, length=vl, width=vw),
            is_visible=vis_target, confidence=1.0 if vis_target else 0.7))

    return PerceptionResult(
        timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid, x=0, y=0, heading=0,
                         velocity=tsp, vx=tsp, vy=0, length=tl, width=tw),
        agents=agents)


def apply_constraint(vid, method, perception):
    """Apply safety constraint: control vehicle ENTIRELY (disable SUMO car-following).

    Vehicles are controlled purely by CoDriving (constant-velocity baseline)
    + safety constraint. SUMO's internal safety is disabled.
    """
    if perception is None:
        return

    ego_speed = perception.ego.velocity

    # Disable SUMO's internal car-following safety for this vehicle
    # speedMode bits: 0=safe speed, 1=max speed, 2=max accel, 3=right-of-way, 4=brake-hard
    # Setting to 0 disables all checks; 6 = respect max speed + max accel only
    try:
        traci.vehicle.setSpeedMode(vid, 6)  # Only respect physical limits, no safety
    except:
        pass

    # CoDriving baseline: maintain current speed (constant velocity)
    target_speed = ego_speed

    # If method is None (NoConstraint): just maintain speed, no safety check
    if method is None:
        try:
            traci.vehicle.setSpeed(vid, max(target_speed, 0))
        except:
            pass
        return

    # Apply safety constraint
    base_wp = np.array([[max(ego_speed, 1.0)*(t+1)*0.5, 0] for t in range(10)])
    try:
        mod_wp, stats = method.constrain_waypoints(base_wp, perception)
    except:
        try:
            traci.vehicle.setSpeed(vid, max(target_speed, 0))
        except:
            pass
        return

    n_threats = stats.get('n_geometric_threats', 0)

    if n_threats > 0:
        # Speed reduction based on modified waypoint
        dist = math.sqrt(mod_wp[0, 0]**2 + mod_wp[0, 1]**2)
        target_speed = min(max(dist / 0.5, 0), ego_speed)

        # Lateral avoidance: lane change if significant y-offset
        lateral_offset = mod_wp[1, 1] if len(mod_wp) > 1 else 0
        if abs(lateral_offset) > 1.5:
            try:
                current_lane = traci.vehicle.getLaneIndex(vid)
                if lateral_offset > 0 and current_lane < 1:
                    traci.vehicle.changeLane(vid, current_lane + 1, 2.0)
                elif lateral_offset < 0 and current_lane > 0:
                    traci.vehicle.changeLane(vid, current_lane - 1, 2.0)
            except:
                pass

    try:
        traci.vehicle.setSpeed(vid, max(target_speed, 0))
    except:
        pass


# =================================================================
# Scenario Runner with Deduplication
# =================================================================

@dataclass
class ScenarioMetrics:
    n_frames: int = 0
    # Deduplicated collisions
    unique_ego_collisions: int = 0  # unique (ego, other) pairs
    secondary_collisions: int = 0   # unique pairs after first
    collision_pairs: set = field(default_factory=set)
    first_collision_step: int = -1
    severities: list = field(default_factory=list)
    # Waypoint metrics
    wp_coll: int = 0
    wp_total: int = 0
    # Detection
    first_warning: int = -1
    n_warnings: int = 0


def run_scenario(net_file, rou_file, method, max_steps=400):
    """Run scenario with collision deduplication."""
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true',
                '--seed', '42',
                '--lanechange.duration', '1.5']

    metrics = ScenarioMetrics()

    try:
        traci.start(sumo_cmd)
        seen_pairs = set()  # (min_id, max_id) pairs already counted

        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            if "ego" not in vids:
                if step > 10: break
                continue

            metrics.n_frames += 1

            # Apply constraint to ALL vehicles
            for vid in vids:
                if method is None:
                    continue
                coop = "coop" if vid == "ego" else ("ego" if vid == "coop" else None)
                perc = build_perception(vid, vids, coop_vid=coop)
                apply_constraint(vid, method, perc)

                # Track ego warnings
                if vid == "ego" and perc:
                    sp = perc.ego.velocity
                    wp = np.array([[max(sp,1.0)*(t+1)*0.5, 0] for t in range(10)])
                    try:
                        _, st = method.constrain_waypoints(wp, perc)
                        if st.get('n_geometric_threats', 0) > 0 or st.get('n_collisions_detected', 0) > 0:
                            if metrics.first_warning < 0:
                                metrics.first_warning = step
                            metrics.n_warnings += 1
                    except:
                        pass

            # Ego waypoint collision check
            ego_perc = build_perception("ego", vids, coop_vid="coop")
            if ego_perc:
                sp = ego_perc.ego.velocity
                wp = np.array([[max(sp,1.0)*(t+1)*0.5, 0] for t in range(10)])
                if method:
                    try: mw, _ = method.constrain_waypoints(wp, ego_perc)
                    except: mw = wp
                else:
                    mw = wp
                for t in range(10):
                    dt = (t+1)*0.5
                    for a in ego_perc.agents:
                        ax = a.state.x + a.state.vx*dt
                        ay = a.state.y + a.state.vy*dt
                        if math.sqrt((mw[t,0]-ax)**2 + (mw[t,1]-ay)**2) < 2.0:
                            metrics.wp_coll += 1; break
                metrics.wp_total += 10

            # COLLISION DEDUPLICATION
            try:
                for col in traci.simulation.getCollisions():
                    # Normalize pair: always (smaller_id, larger_id)
                    pair = tuple(sorted([col.collider, col.victim]))

                    if pair in seen_pairs:
                        continue  # Already counted this pair — skip

                    seen_pairs.add(pair)

                    # Only count ego-involved collisions
                    if "ego" not in pair:
                        continue

                    # Compute severity at first contact
                    try:
                        s1 = traci.vehicle.getSpeed(col.collider) if col.collider in vids else 0
                        s2 = traci.vehicle.getSpeed(col.victim) if col.victim in vids else 0
                        severity = abs(s1 - s2)  # Speed DIFFERENCE (not sum)
                    except:
                        severity = 10.0
                    metrics.severities.append(severity)

                    metrics.unique_ego_collisions += 1
                    if metrics.first_collision_step < 0:
                        metrics.first_collision_step = step
                    else:
                        # This is a collision with a DIFFERENT vehicle → secondary
                        metrics.secondary_collisions += 1
            except:
                pass

        traci.close()
    except Exception as e:
        try: traci.close()
        except: pass

    return metrics


# =================================================================
# Main Experiment
# =================================================================

def main():
    print("=" * 70)
    print("  SUMO v4: Dual-Lane + Lateral Avoidance + Collision Dedup")
    print("=" * 70)
    print("  - 2 lanes per direction (real lane changes possible)")
    print("  - Collision deduplication: same pair counted only once")
    print("  - Secondary = collision with DIFFERENT vehicle after first")
    print("  - All vehicles use CoDriving + safety constraint")

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=0.5),
    }

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_configs = [
        ("t_junction", 15),
        ("crossroads", 15),
    ]

    all_results = {m: {s: [] for s, _ in scenario_configs} for m in methods}
    t0 = time.time()

    for stype, n_sc in scenario_configs:
        print(f"\n  --- {stype} (dual-lane, {n_sc} scenarios × {len(methods)} methods) ---")
        net_file = get_or_create_network(stype, output_dir)

        for sid in range(n_sc):
            rou_file = generate_routes(net_file, stype, sid, n_veh=random.randint(10, 16))
            for mname, method in methods.items():
                r = run_scenario(net_file, rou_file, method)
                all_results[mname][stype].append(r)

            if sid % 5 == 0:
                rn = all_results["NoConstraint"][stype][-1]
                ro = all_results["Ours-Hybrid"][stype][-1]
                print(f"    [{sid}/{n_sc}] NoCon: {rn.unique_ego_collisions} coll "
                      f"| Ours: {ro.unique_ego_collisions} coll, {ro.secondary_collisions} sec")

    # Print results
    print("\n" + "=" * 80)
    print("  RESULTS (dual-lane, deduplicated collisions)")
    print("=" * 80)

    for stype, _ in scenario_configs:
        print(f"\n  [{stype}]")
        print(f"  {'Method':20s} {'UniqColl':>9s} {'2ndColl':>8s} {'2ndRate':>8s} "
              f"{'Severity':>9s} {'WPColl%':>8s} {'DetRate':>8s}")
        print("  " + "-" * 72)

        for mname in methods:
            results = all_results[mname][stype]
            tc = sum(r.unique_ego_collisions for r in results)
            ts = sum(r.secondary_collisions for r in results)
            sr = ts / max(tc, 1)
            sevs = [s for r in results for s in r.severities]
            avg_sev = np.mean(sevs) if sevs else 0
            tw = sum(r.wp_total for r in results)
            twc = sum(r.wp_coll for r in results)
            wr = twc / max(tw, 1)
            # Detection: collision scenarios where warning came before collision
            n_coll_sc = sum(1 for r in results if r.unique_ego_collisions > 0)
            n_det = sum(1 for r in results if r.unique_ego_collisions > 0
                        and r.first_warning >= 0
                        and (r.first_collision_step < 0 or r.first_warning <= r.first_collision_step))
            dr = n_det / max(n_coll_sc, 1)
            print(f"  {mname:20s} {tc:9d} {ts:8d} {sr:7.0%} "
                  f"{avg_sev:8.1f} {wr:7.1%} {dr:7.1%}")

    # Overall
    print(f"\n  {'OVERALL':20s} {'UniqColl':>9s} {'2ndColl':>8s} {'2ndRate':>8s} "
          f"{'Severity':>9s} {'WPColl%':>8s}")
    print("  " + "-" * 65)
    for mname in methods:
        tc=0; ts=0; sevs=[]; tw=0; twc=0
        for stype, _ in scenario_configs:
            for r in all_results[mname][stype]:
                tc += r.unique_ego_collisions; ts += r.secondary_collisions
                sevs.extend(r.severities); tw += r.wp_total; twc += r.wp_coll
        sr = ts/max(tc,1); avg_sev = np.mean(sevs) if sevs else 0; wr = twc/max(tw,1)
        print(f"  {mname:20s} {tc:9d} {ts:8d} {sr:7.0%} "
              f"{avg_sev:8.1f} {wr:7.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

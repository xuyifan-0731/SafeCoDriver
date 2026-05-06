"""Adaptive Safety Controller for SUMO closed-loop.

Improvements over v2:
1. Intersection detection: auto-detect junction proximity → proactive slowdown
2. Graduated braking: limit deceleration to avoid triggering rear-end collisions
3. Leading vehicle monitoring: track front vehicle for sudden brake detection
4. Adaptive following distance: larger gap near high-risk areas

Philosophy: "Smooth safety" — prevent collisions through anticipation, not panic braking.
"""
from __future__ import annotations

import os, sys, math, time, random
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
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
EGO_RANGE = 40.0
COOP_RANGE = 40.0
OTHER_VEH_RANGE = 40.0
SIM_DT = 0.1  # SUMO step length


# =============================================================================
# Adaptive Safety Controller
# =============================================================================

class AdaptiveSafetyController:
    """Wraps a safety method with smooth, adaptive speed control.

    Key features:
    1. Junction proximity detection → proactive speed reduction
    2. Graduated braking (max decel limited) → avoid rear-end triggers
    3. Front vehicle monitoring → react to sudden brakes ahead
    4. Speed history smoothing → no sudden jumps
    """

    def __init__(self, method, name: str = "Adaptive",
                 max_decel: float = 2.5,          # max comfortable decel (m/s²)
                 emergency_decel: float = 5.0,     # emergency decel (m/s²)
                 junction_slowdown: float = 0.7,   # speed factor near junctions
                 junction_detect_dist: float = 15.0,  # meters to start slowing
                 min_following_dist: float = 8.0,  # minimum following gap (m)
                 front_brake_ttc: float = 3.0):    # TTC threshold for front vehicle
        self.method = method
        self.name = name
        self.max_decel = max_decel
        self.emergency_decel = emergency_decel
        self.junction_slowdown = junction_slowdown
        self.junction_detect_dist = junction_detect_dist
        self.min_following_dist = min_following_dist
        self.front_brake_ttc = front_brake_ttc

        # State tracking per vehicle
        self.prev_speeds = defaultdict(lambda: None)  # vid → last target speed
        self.prev_front_speeds = defaultdict(lambda: None)  # vid → front vehicle's last speed

    def compute_target_speed(self, vid: str, perception: PerceptionResult,
                             is_near_junction: bool, current_speed: float) -> float:
        """Compute smooth, safe target speed.

        Strategy:
        - Emergency (method detects imminent collision): allow full braking
        - Preventive (junction approach, following distance): gentle slowdown
        """
        targets = []

        # --- Source 1: Safety constraint method ---
        method_target = self._method_speed(perception, current_speed)
        is_emergency = (method_target is not None and
                        method_target < current_speed * 0.5)

        if method_target is not None:
            targets.append(("method", method_target))

        # --- Source 2: Junction proximity slowdown (preventive only) ---
        # Only slow down if NOT already in emergency
        if is_near_junction and not is_emergency:
            junction_target = current_speed * self.junction_slowdown
            targets.append(("junction", junction_target))

        # --- Source 3: Front vehicle monitoring ---
        front_target = self._front_vehicle_speed(vid, perception, current_speed)
        if front_target is not None:
            targets.append(("front", front_target))

        # --- Source 4: Following distance (preventive) ---
        if not is_emergency:
            gap_target = self._following_distance_speed(perception, current_speed, is_near_junction)
            if gap_target is not None:
                targets.append(("gap", gap_target))

        if not targets:
            return current_speed

        min_target = min(t[1] for t in targets)

        # --- Graduated braking: ONLY for non-emergency ---
        if not is_emergency:
            # Limit deceleration for comfort
            max_drop = self.max_decel * 1.0  # over 1 second
            min_target = max(min_target, current_speed - max_drop)
        else:
            # Emergency: allow full braking but still not instant
            max_drop = self.emergency_decel * 1.0
            min_target = max(min_target, current_speed - max_drop)

        final_target = max(0.0, min(min_target, current_speed))
        self.prev_speeds[vid] = final_target
        return final_target

    def _method_speed(self, perception, current_speed) -> float:
        """Get speed recommendation from the underlying safety method."""
        if self.method is None:
            return None

        base_wp = np.zeros((10, 2))
        for t in range(10):
            base_wp[t, 0] = max(current_speed, 1.0) * (t + 1) * 0.5

        try:
            if hasattr(self.method, 'constrain_waypoints'):
                mod_wp, stats = self.method.constrain_waypoints(base_wp, perception)
                n_threats = stats.get('n_geometric_threats', 0)
                n_det = stats.get('n_collisions_detected', 0)

                if n_threats > 0 or n_det > 0:
                    # Speed from modified first waypoint
                    dist = math.sqrt(mod_wp[0, 0]**2 + mod_wp[0, 1]**2)
                    return max(dist / 0.5, 0.0)
                return None
            else:
                safe = self.method.constrain(perception)
                if safe.mode == ConstraintMode.MINIMUM_HARM:
                    return current_speed * 0.2  # Slow but not stop
                elif safe.mode == ConstraintMode.CONSERVATIVE:
                    return current_speed * 0.6
                return None
        except:
            return None

    def _front_vehicle_speed(self, vid: str, perception: PerceptionResult,
                             current_speed: float) -> float:
        """Monitor front vehicle for sudden braking.

        If front vehicle is decelerating sharply, preemptively slow down.
        """
        # Find agent directly ahead (small lateral offset, positive x)
        front_agent = None
        min_front_dist = 999

        for a in perception.agents:
            s = a.state
            # "In front" = positive x, small |y|
            if s.x > 2.0 and abs(s.y) < 2.5:
                if s.x < min_front_dist:
                    min_front_dist = s.x
                    front_agent = a

        if front_agent is None:
            self.prev_front_speeds[vid] = None
            return None

        front_speed = front_agent.state.velocity
        prev_front = self.prev_front_speeds[vid]
        self.prev_front_speeds[vid] = front_speed

        # Detect sudden braking: front vehicle speed dropped significantly
        if prev_front is not None and prev_front > 1.0:
            decel_rate = (prev_front - front_speed) / SIM_DT
            if decel_rate > 3.0:  # Front is braking hard (>3 m/s²)
                # Match front vehicle speed + safety margin
                return max(front_speed - 1.0, 0.0)

        # TTC-based: if closing in fast
        relative_speed = current_speed - front_speed  # positive = approaching
        if relative_speed > 0 and min_front_dist > 0:
            ttc = min_front_dist / relative_speed
            if ttc < self.front_brake_ttc:
                # Need to slow down to match front speed
                return max(front_speed, 0.0)

        return None

    def _following_distance_speed(self, perception: PerceptionResult,
                                   current_speed: float,
                                   is_near_junction: bool) -> float:
        """Maintain safe following distance.

        Larger gap near junctions for more reaction time.
        """
        min_gap = self.min_following_dist
        if is_near_junction:
            min_gap *= 1.5  # 50% larger gap near junctions

        # Find closest agent in front
        for a in perception.agents:
            s = a.state
            if s.x > 0 and abs(s.y) < 2.0:  # In front, same lane
                gap = s.x - (s.length / 2 + 2.25)  # subtract half-lengths
                if gap < min_gap:
                    # Need to slow down to maintain gap
                    # Target: match front speed, reduced by gap deficit
                    deficit_ratio = max(0, (min_gap - gap) / min_gap)
                    return max(s.velocity * (1 - deficit_ratio * 0.5), 0.0)

        return None


# =============================================================================
# SUMO Infrastructure
# =============================================================================

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
    rou_file = net_file.parent / f"{scenario_type}_v3_{scenario_id}.rou.xml"
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


def is_near_junction(vid: str, net, threshold: float = 15.0) -> bool:
    """Check if vehicle is approaching a MAJOR junction (>=3 incoming edges)."""
    try:
        edge_id = traci.vehicle.getRoadID(vid)
        if edge_id.startswith(':'):  # Already on junction internal edge
            return True
        edge = net.getEdge(edge_id)
        pos_on_edge = traci.vehicle.getLanePosition(vid)
        remaining = edge.getLength() - pos_on_edge
        to_node = edge.getToNode()
        # Only flag as "near junction" for MAJOR junctions (3+ arms)
        if remaining < threshold and len(to_node.getIncoming()) >= 3:
            return True
    except:
        pass
    return False


def build_perception_for_vehicle(target_vid: str, veh_ids: list,
                                  coop_vid: str = None):
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

        cos_h = math.cos(-t_heading)
        sin_h = math.sin(-t_heading)
        rel_x = dx * cos_h - dy * sin_h
        rel_y = dx * sin_h + dy * cos_h
        v_vx = v_speed * math.cos(v_heading)
        v_vy = v_speed * math.sin(v_heading)
        rel_vx = (v_vx - t_vx) * cos_h - (v_vy - t_vy) * sin_h
        rel_vy = (v_vx - t_vx) * sin_h + (v_vy - t_vy) * cos_h

        state = VehicleState(id=vid, x=rel_x, y=rel_y, heading=v_heading-t_heading,
                             velocity=v_speed, vx=rel_vx, vy=rel_vy,
                             length=v_length, width=v_width,
                             vehicle_type='pedestrian' if v_length < 1 else 'car')
        agents.append(Agent(state=state,
                           agent_type=AgentType.PEDESTRIAN if v_length < 1 else AgentType.VEHICLE,
                           is_visible=visible_target, confidence=1.0 if visible_target else 0.7))

    return PerceptionResult(
        timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid, x=0, y=0, heading=0, velocity=t_speed,
                         vx=t_speed, vy=0, length=t_length, width=t_width),
        agents=agents)


# =============================================================================
# Scenario Runner
# =============================================================================

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


def run_scenario(net_file, rou_file, controller: AdaptiveSafetyController,
                 net, max_steps=300):
    """Run scenario with adaptive safety controller on all vehicles."""
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
        ego_coll_count = 0

        for step in range(max_steps):
            traci.simulationStep()
            veh_ids = list(traci.vehicle.getIDList())

            if "ego" not in veh_ids:
                if step > 5:
                    break
                continue

            result.n_frames += 1

            # Apply adaptive safety to ALL vehicles
            for vid in veh_ids:
                if controller.method is None:
                    continue  # NoConstraint: no intervention at all

                # Build perception
                if vid == "ego":
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid="coop")
                elif vid == "coop":
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid="ego")
                else:
                    perception = build_perception_for_vehicle(vid, veh_ids, coop_vid=None)

                if perception is None:
                    continue

                # Check junction proximity
                near_junc = is_near_junction(vid, net)
                current_speed = perception.ego.velocity

                # Get adaptive target speed
                target = controller.compute_target_speed(vid, perception, near_junc, current_speed)

                if target < current_speed:
                    try:
                        # Gradual slowdown over 1.5s (smoother than instant)
                        traci.vehicle.slowDown(vid, max(target, 0), 1.5)
                    except:
                        pass

                    if vid == "ego":
                        if result.first_warning_frame < 0:
                            result.first_warning_frame = step
                        result.n_warning_frames += 1

            # Check ego waypoint collisions
            ego_perc = build_perception_for_vehicle("ego", veh_ids, coop_vid="coop")
            if ego_perc:
                e_speed = ego_perc.ego.velocity
                base_wp = np.array([[max(e_speed,1.0)*(t+1)*0.5, 0] for t in range(10)])
                if controller.method and hasattr(controller.method, 'constrain_waypoints'):
                    try:
                        mod_wp, _ = controller.method.constrain_waypoints(base_wp, ego_perc)
                    except:
                        mod_wp = base_wp
                else:
                    mod_wp = base_wp

                for t in range(10):
                    dt = (t+1)*0.5
                    for a in ego_perc.agents:
                        ax = a.state.x + a.state.vx*dt
                        ay = a.state.y + a.state.vy*dt
                        if math.sqrt((mod_wp[t,0]-ax)**2+(mod_wp[t,1]-ay)**2) < 2.0:
                            result.wp_collisions += 1; break
                result.wp_total += 10

            # Check collisions
            try:
                for col in traci.simulation.getCollisions():
                    if col.collider == "ego" or col.victim == "ego":
                        ego_coll_count += 1
                        try:
                            s1 = traci.vehicle.getSpeed(col.collider) if col.collider in veh_ids else 0
                            s2 = traci.vehicle.getSpeed(col.victim) if col.victim in veh_ids else 0
                            result.collision_severities.append(abs(s1+s2))
                        except:
                            result.collision_severities.append(10.0)
            except:
                pass

        traci.close()
    except Exception as e:
        try: traci.close()
        except: pass

    result.ego_collisions = ego_coll_count
    result.secondary_collisions = max(ego_coll_count - 1, 0)
    return result


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("  SUMO v3: Adaptive Safety Controller")
    print("  (junction detection + graduated braking + front monitoring)")
    print("="*70)

    import torch
    from coop_safety.learned.collision_network import CollisionPredictionNetwork

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu', weights_only=False)['model'])
    v1.eval()

    # Create base methods
    base_methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(detector_model=v1,
                                               base_margin_visible=2.5,
                                               base_margin_invisible=4.0,
                                               detection_threshold=0.3),
    }

    # Wrap with adaptive controller
    controllers = {}
    for name, method in base_methods.items():
        controllers[name] = AdaptiveSafetyController(
            method=method, name=name,
            max_decel=2.5, emergency_decel=5.0,
            junction_slowdown=0.6, junction_detect_dist=25.0,
            min_following_dist=8.0, front_brake_ttc=3.0)

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_configs = [
        ("t_junction", 15),
        ("crossroads", 15),
        ("blind_pedestrian", 10),
    ]

    all_results = {m: {s: [] for s, _ in scenario_configs} for m in controllers}
    t0 = time.time()

    for stype, n_sc in scenario_configs:
        print(f"\n  --- {stype} ({n_sc} scenarios × {len(controllers)} methods) ---")
        net_file = generate_network(stype, output_dir)
        net = sumolib.net.readNet(str(net_file))

        for sid in range(n_sc):
            rou_file = generate_route_file(net_file, stype, sid,
                                           n_vehicles=random.randint(6, 10))
            for mname, ctrl in controllers.items():
                result = run_scenario(net_file, rou_file, ctrl, net)
                all_results[mname][stype].append(result)

            if sid % 5 == 0:
                r_ours = all_results["Ours-Hybrid"][stype][-1]
                r_nc = all_results["NoConstraint"][stype][-1]
                print(f"    [{sid}/{n_sc}] NoCon: coll={r_nc.ego_collisions} "
                      f"Ours: coll={r_ours.ego_collisions} sec={r_ours.secondary_collisions}")

    # Print results
    print("\n" + "="*80)
    print("  RESULTS: Adaptive Safety Controller (v3)")
    print("  (junction slowdown + graduated braking + front monitoring)")
    print("="*80)

    for stype, n_sc in scenario_configs:
        print(f"\n  [{stype}] {n_sc} scenarios")
        print(f"  {'Method':20s} {'EgoColl':>8s} {'2ndColl':>8s} {'2ndRate':>8s} "
              f"{'Severity':>9s} {'WPColl%':>8s}")
        print("  " + "-"*65)

        for mname in controllers:
            results = all_results[mname][stype]
            tc = sum(r.ego_collisions for r in results)
            ts = sum(r.secondary_collisions for r in results)
            sr = ts / max(tc, 1)
            sevs = [s for r in results for s in r.collision_severities]
            avg_sev = np.mean(sevs) if sevs else 0
            tw = sum(r.wp_total for r in results)
            twc = sum(r.wp_collisions for r in results)
            wr = twc / max(tw, 1)
            print(f"  {mname:20s} {tc:8d} {ts:8d} {sr:7.0%} "
                  f"{avg_sev:8.1f} {wr:7.1%}")

    # Overall
    print("\n" + "="*80)
    print("  OVERALL")
    print("="*80)
    print(f"  {'Method':20s} {'EgoColl':>8s} {'2ndColl':>8s} {'2ndRate':>8s} "
          f"{'Severity':>9s} {'WPColl%':>8s}")
    print("  " + "-"*65)

    for mname in controllers:
        tc=0;ts=0;sevs=[];tw=0;twc=0
        for stype, _ in scenario_configs:
            for r in all_results[mname][stype]:
                tc+=r.ego_collisions; ts+=r.secondary_collisions
                sevs.extend(r.collision_severities); tw+=r.wp_total; twc+=r.wp_collisions
        sr=ts/max(tc,1); avg_sev=np.mean(sevs) if sevs else 0; wr=twc/max(tw,1)
        print(f"  {mname:20s} {tc:8d} {ts:8d} {sr:7.0%} "
              f"{avg_sev:8.1f} {wr:7.1%}")

    # Comparison with v2
    print("\n  [Comparison with v2 (no adaptive control)]")
    print("  v2 Ours-Hybrid: 10 collisions, 8 secondary (80%), severity 11.0")
    print(f"  v3 Ours-Hybrid: see above")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

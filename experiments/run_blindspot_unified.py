"""Blind-spot SUMO scenarios with explicit occlusion.

4 blind-spot scenario types:
  BS1: Front-right U-turn (前方右侧车辆掉头)
       Attacker turns from oncoming lane into ego's lane while a blocker
       (large slow vehicle) hides it from ego's view.
  BS2: Pedestrian-like sudden crossing (前方车辆右侧出现横穿马路的行人)
       Slow agent emerges from behind a parked-style blocker on the right,
       crossing ego's path.
  BS3: T-junction blind corner (盲区交叉路口)
       Attacker from perpendicular street; building/blocker at corner
       hides them from ego until last second.
  BS4: Right-side hidden merge (右侧盲区并道)
       Adjacent-street vehicle merges left into ego's path; truck on
       right blocks ego's view.

For each scenario type:
  - "att_*" vehicle IDs are attackers (source of threat)
  - "blk_*" vehicle IDs are blockers (create occlusion)
  - Visibility logic: an attacker is invisible to ego if a blocker
    is geometrically between ego and the attacker (line-of-sight check).

Baseline list:
  - NoCon-egoonly : ego sees only its own range, no V2X, no constraint
  - NoCon-coop    : ego + coop V2X perception, no constraint
  - RSS-coop, UniE2EV2X-coop, MAP-coop, RiskMM-coop
  - Ours-Hybrid (thr ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.50})
  - Ours-Hybrid+AND (thr ∈ {0.25, 0.30, 0.35, 0.40})
"""
from __future__ import annotations
import sys, os, math, time, random, logging, re
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

sys.path.insert(0, str(Path(__file__).parent.parent))
from coop_safety.interface import (PerceptionResult, VehicleState, Agent,
                                    AgentType, ConstraintMode)
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
EGO_RANGE = 50.0
COOP_RANGE = 50.0
BLOCKER_RADIUS = 4.0  # meters; if blocker is within this distance of LOS, it occludes


# =================================================================
# Network generation
# =================================================================

def get_or_create_network(net_name: str) -> Path:
    """Networks needed:
       - cross_1lane: 4-arm crossroads, 1 lane (BS3)
       - cross_2lane: 4-arm crossroads, 2 lanes (BS1, BS4)
       - long_2lane:  straight 2-lane road (BS2)
    """
    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)
    net_file = output_dir / f"{net_name}.net.xml"
    if net_file.exists():
        return net_file

    if net_name == 'cross_1lane':
        os.system(f"{NETGEN} --spider --spider.arm-number 4 --spider.space-radius 80 "
                  f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 1 "
                  f"-o {net_file} 2>/dev/null")
    elif net_name == 'cross_2lane':
        os.system(f"{NETGEN} --spider --spider.arm-number 4 --spider.space-radius 80 "
                  f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 2 "
                  f"-o {net_file} 2>/dev/null")
    elif net_name == 'long_2lane':
        # Use spider with 2 arms ≈ straight road
        os.system(f"{NETGEN} --grid --grid.x-number 3 --grid.y-number 1 "
                  f"--grid.x-length 100 --grid.y-length 100 "
                  f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 2 "
                  f"-o {net_file} 2>/dev/null")
    else:
        raise ValueError(f"Unknown network: {net_name}")
    return net_file


# =================================================================
# Scenario generators
# =================================================================

def _routes_through_center(net) -> dict:
    """Find center node and return {in_edge_id: [out_edge_ids]}."""
    center = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3:
            center = n; break
    if center is None:
        return {}
    routes = {}
    for ei in center.getIncoming():
        outs = []
        for eo in ei.getOutgoing():
            outs.append(eo.getID())
        routes[ei.getID()] = outs
    return routes


def _inject_blocker(rou_path: Path, blocker_xml: str) -> Path:
    """Insert blocker XML lines just before </routes> closing tag.

    Also injects 'blocker' vType if missing.
    """
    text = open(rou_path).read()
    # Inject blocker vType after <routes> if not present
    if 'id="blocker"' not in text:
        blocker_vtype = (
            '    <vType id="blocker" length="8.0" width="2.5" maxSpeed="6.0" '
            'accel="2.0" decel="6.0" sigma="0.0" minGap="0.5" tau="0.5"/>\n'
            '    <vType id="pedestrian" length="0.6" width="0.6" maxSpeed="2.5" '
            'accel="2.0" decel="3.0" sigma="0.0" minGap="0.1" tau="0.1"/>\n')
        text = text.replace('<routes>\n', '<routes>\n' + blocker_vtype, 1)
    # Insert blocker vehicle
    new_text = text.replace('</routes>', blocker_xml + '\n</routes>')
    # Sort vehicles by depart
    pre, mid = new_text.split('<routes>', 1)
    parts = re.split(r'(?=<vehicle\s)', mid)
    pre_vehicles = parts[0]  # vTypes section
    veh_parts = parts[1:]
    # Last veh part contains </routes> at end
    if veh_parts:
        last = veh_parts[-1]
        last_clean, end_part = last.rsplit('</routes>', 1)
        veh_parts[-1] = last_clean
    else:
        end_part = '</routes>'
    def get_dep(v):
        m = re.search(r'depart="([\d.]+)"', v)
        return float(m.group(1)) if m else 0.0
    veh_parts.sort(key=get_dep)
    new_text = pre + '<routes>' + pre_vehicles + ''.join(veh_parts) + '</routes>' + end_part.split('</routes>')[-1] if '</routes>' in end_part else pre + '<routes>' + pre_vehicles + ''.join(veh_parts) + '</routes>'
    # Simpler: rebuild cleanly
    new_text = pre + '<routes>' + pre_vehicles + ''.join(veh_parts) + '</routes>\n'
    open(rou_path, 'w').write(new_text)
    return rou_path


def gen_BS_via_forced(net_file: Path, sid: int, seed: int,
                      conflict_type: int, has_blocker: bool = True,
                      bs_label: str = 'BS') -> Path:
    """Use the proven generate_forced_conflict and inject a blocker.
    conflict_type: 1=head-on, 3=rear-end (these produce reliable collisions).
    """
    from experiments.run_forced_conflict_and_fa import generate_forced_conflict

    base = generate_forced_conflict(net_file, conflict_type, f"{bs_label}base", seed=seed)
    # Copy to bs-specific path so we can inject blocker without altering base
    out_path = net_file.parent / f"{bs_label}_{sid}_{seed}.rou.xml"
    text = open(base).read()
    open(out_path, 'w').write(text)
    if not has_blocker:
        return out_path

    # Build blocker XML
    net = sumolib.net.readNet(str(net_file))
    routes = _routes_through_center(net)
    in_ids = list(routes.keys())
    ego_in = in_ids[0]
    ego_out = routes[ego_in][0]
    att_in = (in_ids[2] if conflict_type == 1 else ego_in) if len(in_ids) > 2 else in_ids[0]

    if bs_label == 'BS1':  # blocker between ego and head-on attacker
        # Place blocker on ego's lane ahead, slow
        blocker_xml = (
            f'    <vehicle id="blk_truck" type="blocker" depart="0.1" departSpeed="5">'
            f'\n        <route edges="{ego_in} {ego_out}"/>\n    </vehicle>')
    elif bs_label == 'BS2':  # parked-style blocker for ped scenario
        blocker_xml = (
            f'    <vehicle id="blk_park" type="blocker" depart="0.1" departSpeed="2">'
            f'\n        <route edges="{ego_in} {ego_out}"/>\n    </vehicle>')
    elif bs_label == 'BS3':  # corner blocker on attacker's edge
        blocker_xml = (
            f'    <vehicle id="blk_corner" type="blocker" depart="0.1" departSpeed="0">'
            f'\n        <route edges="{att_in} {routes[att_in][0]}"/>\n    </vehicle>')
    else:  # BS4 right-side merge
        if 'cross_2lane' in str(net_file):
            blocker_xml = (
                f'    <vehicle id="blk_truck" type="blocker" depart="0.1" departSpeed="5" departLane="1">'
                f'\n        <route edges="{ego_in} {ego_out}"/>\n    </vehicle>')
        else:
            blocker_xml = (
                f'    <vehicle id="blk_truck" type="blocker" depart="0.1" departSpeed="5">'
                f'\n        <route edges="{ego_in} {ego_out}"/>\n    </vehicle>')

    return _inject_blocker(out_path, blocker_xml)


def gen_BS1_uturn(net_file, sid, seed):
    return gen_BS_via_forced(net_file, sid, seed, conflict_type=1, bs_label='BS1')

def gen_BS2_pedestrian(net_file, sid, seed):
    # Use head-on with parked blocker; replace att_speed for ped behavior
    p = gen_BS_via_forced(net_file, sid, seed, conflict_type=1, bs_label='BS2')
    # Replace attacker speed to "pedestrian" behavior
    text = open(p).read()
    text = text.replace('type="attacker"', 'type="pedestrian"', 1)
    text = re.sub(r'(id="attacker"[^>]*departSpeed=)"[\d.]+"', r'\1"2.5"', text, count=1)
    open(p, 'w').write(text)
    return p

def gen_BS3_blindcorner(net_file, sid, seed):
    # Perpendicular crossing — use ct=1 with corner blocker on attacker's edge
    return gen_BS_via_forced(net_file, sid, seed, conflict_type=1, bs_label='BS3')

def gen_BS4_hidden_merge(net_file, sid, seed):
    return gen_BS_via_forced(net_file, sid, seed, conflict_type=3, bs_label='BS4')


def _write_routes(out_path: Path, vehs: list) -> Path:
    # Sort by depart
    def get_depart(line):
        m = re.search(r'depart="([\d.]+)"', line)
        return float(m.group(1)) if m else 0.0
    vehs.sort(key=get_depart)
    content = f"""<?xml version="1.0"?>
<routes>
    <vType id="ego_car" length="4.5" width="1.8" maxSpeed="13.5" accel="3.5" decel="6.0"
           sigma="0.0" minGap="0.2" tau="0.2" speedDev="0.0"/>
    <vType id="attacker" length="4.5" width="1.8" maxSpeed="13.5" accel="5.0" decel="4.0"
           sigma="0.0" minGap="0.1" tau="0.1" speedDev="0.0" impatience="1.0"/>
    <vType id="blocker" length="8.0" width="2.5" maxSpeed="6.0" accel="2.0" decel="6.0"
           sigma="0.0" minGap="0.5" tau="0.5"/>
    <vType id="pedestrian" length="0.6" width="0.6" maxSpeed="2.5" accel="2.0" decel="3.0"
           sigma="0.0" minGap="0.1" tau="0.1"/>
    <vType id="car" length="4.5" width="1.8" maxSpeed="13.5" accel="3.0" decel="5.0"
           sigma="0.4" minGap="0.5" tau="0.4" speedDev="0.2"/>
{chr(10).join(vehs)}
</routes>"""
    open(out_path, 'w').write(content)
    return out_path


# =================================================================
# Perception with line-of-sight occlusion
# =================================================================

def get_veh(vid):
    try:
        x, y = traci.vehicle.getPosition(vid)
        sp = traci.vehicle.getSpeed(vid)
        ang = traci.vehicle.getAngle(vid)
        return (x, y, sp, math.radians(90 - ang),
                traci.vehicle.getLength(vid), traci.vehicle.getWidth(vid))
    except Exception:
        return None


def is_los_blocked(ego_pos, target_pos, blockers_pos, radius=BLOCKER_RADIUS):
    """Check if line of sight from ego to target is blocked by any blocker.
    blockers_pos: list of (x, y) for blocker positions.
    Returns True if blocked (target invisible to ego).
    """
    ex, ey = ego_pos
    tx, ty = target_pos
    seg_dx = tx - ex
    seg_dy = ty - ey
    seg_len = math.sqrt(seg_dx**2 + seg_dy**2)
    if seg_len < 0.1:
        return False
    for bx, by in blockers_pos:
        # Vector from ego to blocker
        bdx = bx - ex
        bdy = by - ey
        # Project onto ego->target segment
        t = (bdx * seg_dx + bdy * seg_dy) / (seg_len * seg_len)
        if t < 0 or t > 1:
            continue  # Blocker not between ego and target
        # Distance from blocker to segment
        proj_x = ex + t * seg_dx
        proj_y = ey + t * seg_dy
        dist_to_seg = math.sqrt((bx - proj_x)**2 + (by - proj_y)**2)
        if dist_to_seg < radius:
            return True
    return False


def build_perception_los(target_vid, vids, coop_vid=None, ego_only=False):
    """Build perception with line-of-sight occlusion.

    ego_only=True: ignores coop entirely (no V2X baseline).
    Vehicles with id starting "blk_" are blockers.
    """
    ts = get_veh(target_vid)
    if ts is None: return None
    tx, ty, tsp, th, tl, tw = ts
    tvx, tvy = tsp * math.cos(th), tsp * math.sin(th)

    cp = None
    if not ego_only and coop_vid and coop_vid in vids:
        ci = get_veh(coop_vid)
        if ci: cp = (ci[0], ci[1])

    # Collect blocker positions
    blockers_pos = []
    for v in vids:
        if v.startswith('blk_'):
            bi = get_veh(v)
            if bi: blockers_pos.append((bi[0], bi[1]))

    agents = []
    for v in vids:
        if v in (target_vid, coop_vid): continue
        vi = get_veh(v)
        if vi is None: continue
        vx, vy, vs, vh, vl, vw = vi

        dx, dy = vx - tx, vy - ty
        de = math.sqrt(dx**2 + dy**2)
        # Visibility from target_vid (ego)
        vis_ego_dist = de <= EGO_RANGE
        # LOS: blocked if any blocker between ego and v
        vis_ego_los = not is_los_blocked((tx, ty), (vx, vy), blockers_pos)
        vis_target = vis_ego_dist and vis_ego_los

        # Visibility from coop
        vis_coop = False
        if cp:
            d_coop = math.sqrt((vx - cp[0])**2 + (vy - cp[1])**2)
            # Coop's own LOS: blocked by blockers
            vis_coop_los = not is_los_blocked(cp, (vx, vy), blockers_pos)
            vis_coop = (d_coop <= COOP_RANGE) and vis_coop_los

        if not vis_target and not vis_coop: continue

        ch, sh = math.cos(-th), math.sin(-th)
        rx, ry = dx * ch - dy * sh, dx * sh + dy * ch
        vvx, vvy = vs * math.cos(vh), vs * math.sin(vh)
        rvx = (vvx - tvx) * ch - (vvy - tvy) * sh
        rvy = (vvx - tvx) * sh + (vvy - tvy) * ch

        agents.append(Agent(state=VehicleState(id=v, x=rx, y=ry, heading=vh - th,
                            velocity=vs, vx=rvx, vy=rvy, length=vl, width=vw),
                            is_visible=vis_target,
                            confidence=1.0 if vis_target else 0.7))

    return PerceptionResult(timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid, x=0, y=0, heading=0, velocity=tsp,
                         vx=tsp, vy=0, length=tl, width=tw), agents=agents)


# =================================================================
# Run scenario with full metrics
# =================================================================

@dataclass
class FullMetrics:
    n_frames: int = 0
    # Collisions (deduplicated)
    unique_ego_coll: int = 0
    secondary_coll: int = 0
    severities: list = field(default_factory=list)
    first_collision: int = -1
    # Detection / warning
    first_warning: int = -1
    n_warning_frames: int = 0
    # Frame-level: dangerous (TTC<3s) frames where method warned
    n_dangerous_frames: int = 0
    n_warned_dangerous: int = 0
    # Frame-level FA: non-dangerous frames where method warned
    n_safe_frames: int = 0
    n_warned_safe: int = 0
    # Waypoints
    wp_coll: int = 0
    wp_total: int = 0
    # Mod rate
    n_modified_frames: int = 0
    # Bookkeeping
    is_collision_scenario: bool = False  # actual collision happened


def estimate_ttc_to_attacker(ego_perc):
    """Estimate min TTC from ego to any agent."""
    if not ego_perc or not ego_perc.agents:
        return 999
    min_ttc = 999
    ego_speed = ego_perc.ego.velocity
    for a in ego_perc.agents:
        s = a.state
        dist = math.sqrt(s.x**2 + s.y**2)
        if dist < 0.5:
            return 0  # Already collided
        # Closing speed: project rel velocity onto ego->agent direction
        rel_vx = s.vx - ego_speed  # actually vx is already relative
        rel_vy = s.vy
        # Approach rate
        if dist > 0.1:
            approach = -(s.x * s.vx + s.y * s.vy) / dist
            if approach > 0.1:
                ttc = dist / approach
                if ttc < min_ttc:
                    min_ttc = ttc
    return min_ttc


def run_scenario(net_file, rou_file, method, ego_uses_coop=True,
                 max_steps=300, sumo_seed=42):
    """Run blind-spot scenario.

    ego_uses_coop=False simulates "no V2X" baseline.
    """
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true',
                '--seed', str(sumo_seed),
                '--lanechange.duration', '1.5']
    m = FullMetrics()
    try:
        traci.start(sumo_cmd)
        seen_pairs = set()

        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            if "ego" not in vids:
                if step > 5: break
                continue
            m.n_frames += 1

            # Match working run_sumo_forced behavior:
            # - Only ego and attackers get speedMode override
            # - Other vehicles (coop, distractors, blockers) keep default safety
            for v in vids:
                if v == "attacker" or v.startswith('att_'):
                    try: traci.vehicle.setSpeedMode(v, 0)
                    except Exception: pass
                elif v == "ego":
                    try: traci.vehicle.setSpeedMode(v, 6)
                    except Exception: pass

            # Build ego perception (with or without coop)
            coop_id = "coop" if ego_uses_coop else None
            ep = build_perception_los("ego", vids, coop_vid=coop_id,
                                       ego_only=not ego_uses_coop)

            # Track ego's TTC for frame-level danger annotation
            ttc = estimate_ttc_to_attacker(ep)
            is_dangerous_frame = ttc < 3.0

            ego_warned = False
            target_speed = ep.ego.velocity if ep else 11.0

            if ep:
                if method is None:
                    # NoConstraint baseline: maintain current speed (CoDriving)
                    target_speed = ep.ego.velocity
                else:
                    bw = np.array([[max(ep.ego.velocity, 1.0) * (t + 1) * 0.5, 0]
                                  for t in range(10)])
                    try:
                        if hasattr(method, 'constrain_waypoints'):
                            mw, stats = method.constrain_waypoints(bw, ep)
                            fired = (stats.get('n_geometric_threats', 0) > 0
                                     or stats.get('n_collisions_detected', 0) > 0
                                     or stats.get('n_modifications', 0) > 0)
                            if fired:
                                ego_warned = True
                                # P0 (260511): use explicit target_speed_factor if provided
                                tsf = stats.get('target_speed_factor', None)
                                if tsf is not None:
                                    target_speed = ep.ego.velocity * tsf
                                else:
                                    # Fallback: derive from modified waypoint distance
                                    d = math.sqrt(mw[0, 0]**2 + mw[0, 1]**2)
                                    target_speed = min(max(d / 0.5, 0), ep.ego.velocity)
                                # Lateral
                                if len(mw) > 1 and abs(mw[1, 1]) > 1.5:
                                    try:
                                        cl = traci.vehicle.getLaneIndex("ego")
                                        # Get lane count from current edge
                                        cur_road = traci.vehicle.getRoadID("ego")
                                        n_lanes = 1
                                        try:
                                            n_lanes = traci.edge.getLaneNumber(cur_road)
                                        except Exception:
                                            pass
                                        if mw[1, 1] > 0 and cl < n_lanes - 1:
                                            traci.vehicle.changeLane("ego", cl + 1, 2.0)
                                        elif mw[1, 1] < 0 and cl > 0:
                                            traci.vehicle.changeLane("ego", cl - 1, 2.0)
                                    except Exception: pass
                        else:
                            safe = method.constrain(ep)
                            if safe.mode == ConstraintMode.MINIMUM_HARM:
                                ego_warned = True
                                target_speed = max(ep.ego.velocity * 0.2, 0)
                            elif safe.mode == ConstraintMode.CONSERVATIVE:
                                ego_warned = True
                                target_speed = max(ep.ego.velocity * 0.6, 0)
                    except Exception as e:
                        logger.warning(f"method failed: {e}")

            # Apply ego speed (always, even for NoConstraint)
            try:
                traci.vehicle.setSpeed("ego", max(target_speed, 0))
            except Exception: pass

            # Attendre les autres véhicules normalement (no constraint applied)
            # Coop, distractors, blockers all use SUMO default car-following
            # already disabled by setSpeedMode(6); maintain their own speeds.

            # WPColl check on ego
            if ep:
                wp = np.array([[max(ep.ego.velocity, 1.0) * (t + 1) * 0.5, 0]
                              for t in range(10)])
                if method and hasattr(method, 'constrain_waypoints'):
                    try: mw_check, _ = method.constrain_waypoints(wp, ep)
                    except Exception: mw_check = wp
                else: mw_check = wp
                for t in range(10):
                    dt = (t + 1) * 0.5
                    for a in ep.agents:
                        ax = a.state.x + a.state.vx * dt
                        ay = a.state.y + a.state.vy * dt
                        if math.sqrt((mw_check[t, 0] - ax)**2 +
                                     (mw_check[t, 1] - ay)**2) < 2.0:
                            m.wp_coll += 1; break
                m.wp_total += 10

            # Track frame-level danger / warning
            if is_dangerous_frame:
                m.n_dangerous_frames += 1
                if ego_warned: m.n_warned_dangerous += 1
            else:
                m.n_safe_frames += 1
                if ego_warned: m.n_warned_safe += 1

            if ego_warned:
                m.n_warning_frames += 1
                m.n_modified_frames += 1
                if m.first_warning < 0:
                    m.first_warning = step

            # Collisions
            try:
                for col in traci.simulation.getCollisions():
                    pair = tuple(sorted([col.collider, col.victim]))
                    if pair in seen_pairs: continue
                    seen_pairs.add(pair)
                    if "ego" not in pair: continue
                    try:
                        s1 = traci.vehicle.getSpeed(col.collider) if col.collider in vids else 0
                        s2 = traci.vehicle.getSpeed(col.victim) if col.victim in vids else 0
                        m.severities.append(abs(s1 - s2))
                    except Exception:
                        m.severities.append(10.0)
                    m.unique_ego_coll += 1
                    if m.first_collision < 0: m.first_collision = step
                    else: m.secondary_coll += 1
            except Exception: pass

        traci.close()
    except Exception as e:
        logger.warning(f"SUMO error: {e}")
        try: traci.close()
        except Exception: pass

    m.is_collision_scenario = m.unique_ego_coll > 0
    return m


# =================================================================
# Main eval
# =================================================================

@dataclass
class AggResult:
    n_runs: int = 0
    coll_runs: int = 0  # # of runs with collision
    ego_coll_total: int = 0
    secondary_total: int = 0
    severities: list = field(default_factory=list)
    # Per-frame
    n_frames: int = 0
    n_dangerous_frames: int = 0
    n_warned_dangerous: int = 0
    n_safe_frames: int = 0
    n_warned_safe: int = 0
    # Scen-level
    n_detect_scen: int = 0  # collision scenarios where method warned BEFORE collision
    n_fa_scen: int = 0      # no-collision scenarios where method warned at all
    n_normal_runs: int = 0
    early_warnings: list = field(default_factory=list)
    wp_coll: int = 0
    wp_total: int = 0
    n_modified_frames: int = 0


def main():
    print("=" * 70)
    print("  Blind-Spot SUMO + Full-Metrics Eval (260511)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    # Method/variant configurations
    # Each entry: (name, factory, ego_uses_coop)
    # Factory creates a fresh instance per scenario (to reset stateful methods)
    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5, base_margin_invisible=4.0)

    method_configs = [
        ("NoCon-egoonly",  lambda: None,  False),  # No V2X baseline
        ("NoCon-coop",     lambda: None,  True),
        ("RSS-coop",       lambda: RSSOnly(),  True),
        ("UniE2EV2X-coop", lambda: UniE2EV2XSafety(safety_threshold=3.0), True),
        ("MAP-coop",       lambda: MAPSafety(min_clearance=0.5), True),
        ("RiskMM-coop",    lambda: RiskMMSafety(v_max=20.0), True),
    ]
    # Hybrid variants at different thresholds
    for thr in [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        method_configs.append(
            (f"Hybrid-thr{thr:.2f}",
             lambda thr=thr: HybridSafetyConstraint(detection_threshold=thr, **base_kwargs),
             True))
    # Hybrid+AND at different thresholds
    for thr in [0.25, 0.30, 0.35, 0.40]:
        def make_and(thr=thr):
            base = HybridSafetyConstraint(detection_threshold=thr, **base_kwargs)
            return HybridWithGeometricAND(base)
        method_configs.append((f"Hybrid+AND-thr{thr:.2f}", make_and, True))

    # Scenarios: focus on reliable collision-producing setups
    cross_1l = get_or_create_network('cross_1lane')
    cross_2l = get_or_create_network('cross_2lane')

    scenarios = [
        ('BS1_uturn', cross_1l, gen_BS1_uturn),
        ('BS3_corner', cross_1l, gen_BS3_blindcorner),
        ('BS4_merge', cross_2l, gen_BS4_hidden_merge),
    ]
    seeds = [42, 123, 789]
    n_sids = 4  # 4 scenario variants per seed

    # Results: (method_name, scenario_name) -> AggResult
    results = defaultdict(AggResult)

    t0 = time.time()
    for sname, net_file, gen_fn in scenarios:
        print(f"\n  --- {sname} ---")
        for seed in seeds:
            for sid in range(n_sids):
                rou_file = gen_fn(net_file, sid, seed)
                for mname, factory, uses_coop in method_configs:
                    method = factory()
                    r = run_scenario(net_file, rou_file, method,
                                     ego_uses_coop=uses_coop, sumo_seed=seed)
                    agg = results[(mname, sname)]
                    agg.n_runs += 1
                    agg.ego_coll_total += r.unique_ego_coll
                    agg.secondary_total += r.secondary_coll
                    agg.severities.extend(r.severities)
                    agg.n_frames += r.n_frames
                    agg.n_dangerous_frames += r.n_dangerous_frames
                    agg.n_warned_dangerous += r.n_warned_dangerous
                    agg.n_safe_frames += r.n_safe_frames
                    agg.n_warned_safe += r.n_warned_safe
                    agg.wp_coll += r.wp_coll; agg.wp_total += r.wp_total
                    agg.n_modified_frames += r.n_modified_frames
                    if r.is_collision_scenario:
                        agg.coll_runs += 1
                        # Det = warning before collision in collision scenarios
                        if r.first_warning >= 0 and r.first_warning <= r.first_collision:
                            agg.n_detect_scen += 1
                            agg.early_warnings.append(r.first_collision - r.first_warning)
                    else:
                        agg.n_normal_runs += 1
                        if r.first_warning >= 0:
                            agg.n_fa_scen += 1
        # Print partial
        for mname, _, _ in method_configs:
            agg = results[(mname, sname)]
            cr = agg.coll_runs / max(agg.n_runs, 1)
            print(f"    {mname:25s}: {agg.coll_runs}/{agg.n_runs} coll-runs ({cr:.0%})")

    # =================================================================
    # Print full results per scenario
    # =================================================================
    sevs_min = 1e-9
    for sname, _, _ in scenarios:
        print(f"\n{'=' * 70}")
        print(f"  Scenario: {sname}")
        print('=' * 70)
        print(f"  {'Method':25s} {'CollRate':>9s} {'2ndC':>5s} {'Sev':>5s} "
              f"{'Det(s)':>7s} {'Det(f)':>7s} {'Early':>6s} "
              f"{'FA(s)':>6s} {'FA(f)':>6s} {'WPC%':>6s} {'Mod%':>5s}")
        print('  ' + '-' * 105)
        for mname, _, _ in method_configs:
            agg = results[(mname, sname)]
            coll_rate = agg.coll_runs / max(agg.n_runs, 1)
            det_s = agg.n_detect_scen / max(agg.coll_runs, 1)
            det_f = agg.n_warned_dangerous / max(agg.n_dangerous_frames, 1)
            fa_s = agg.n_fa_scen / max(agg.n_normal_runs, 1)
            fa_f = agg.n_warned_safe / max(agg.n_safe_frames, 1)
            sev = np.mean(agg.severities) if agg.severities else 0
            early = np.mean(agg.early_warnings) if agg.early_warnings else 0
            wpc = agg.wp_coll / max(agg.wp_total, 1)
            mod = agg.n_modified_frames / max(agg.n_frames, 1)
            sec_rate = agg.secondary_total / max(agg.ego_coll_total, 1)
            print(f"  {mname:25s} {coll_rate:8.0%} {agg.secondary_total:5d} "
                  f"{sev:5.2f} {det_s:6.0%} {det_f:6.0%} {early:5.1f} "
                  f"{fa_s:5.0%} {fa_f:5.0%} {wpc:5.1%} {mod:4.0%}")

    # Overall ranking
    print(f"\n{'=' * 70}")
    print("  OVERALL (averaged across all blind-spot scenarios)")
    print('=' * 70)
    print(f"  {'Method':25s} {'CollRate':>9s} {'2ndC':>5s} {'Sev':>5s} "
          f"{'Det(s)':>7s} {'Det(f)':>7s} {'Early':>6s} "
          f"{'FA(s)':>6s} {'FA(f)':>6s} {'WPC%':>6s} {'Mod%':>5s}")
    print('  ' + '-' * 105)
    summary = []
    for mname, _, _ in method_configs:
        runs = 0; cr_runs = 0; ec = 0; sec = 0; sevs = []
        nf = 0; nd = 0; nw_d = 0; nsf = 0; nw_s = 0
        det_s_n = 0; det_s_d = 0
        fa_s_n = 0; fa_s_d = 0
        ew = []
        wpc = 0; wpt = 0; mn = 0
        for sname, _, _ in scenarios:
            agg = results[(mname, sname)]
            runs += agg.n_runs; cr_runs += agg.coll_runs
            ec += agg.ego_coll_total; sec += agg.secondary_total
            sevs.extend(agg.severities)
            nf += agg.n_frames; nd += agg.n_dangerous_frames
            nw_d += agg.n_warned_dangerous; nsf += agg.n_safe_frames
            nw_s += agg.n_warned_safe
            det_s_n += agg.n_detect_scen; det_s_d += agg.coll_runs
            fa_s_n += agg.n_fa_scen; fa_s_d += agg.n_normal_runs
            ew.extend(agg.early_warnings)
            wpc += agg.wp_coll; wpt += agg.wp_total; mn += agg.n_modified_frames
        coll_rate = cr_runs / max(runs, 1)
        det_s = det_s_n / max(det_s_d, 1)
        det_f = nw_d / max(nd, 1)
        fa_s = fa_s_n / max(fa_s_d, 1)
        fa_f = nw_s / max(nsf, 1)
        sev = np.mean(sevs) if sevs else 0
        e_avg = np.mean(ew) if ew else 0
        wpc_r = wpc / max(wpt, 1)
        mod = mn / max(nf, 1)
        sec_rate = sec / max(ec, 1)
        summary.append({
            'name': mname, 'coll_rate': coll_rate, 'sec': sec, 'sec_rate': sec_rate,
            'sev': sev, 'det_s': det_s, 'det_f': det_f, 'early': e_avg,
            'fa_s': fa_s, 'fa_f': fa_f, 'wpc': wpc_r, 'mod': mod})
        print(f"  {mname:25s} {coll_rate:8.0%} {sec:5d} {sev:5.2f} "
              f"{det_s:6.0%} {det_f:6.0%} {e_avg:5.1f} "
              f"{fa_s:5.0%} {fa_f:5.0%} {wpc_r:5.1%} {mod:4.0%}")

    # Ranking by user priority: WPColl% > 2nd-coll > DetRate > FA > Severity > others
    print(f"\n{'=' * 70}")
    print("  RANKING BY USER PRIORITY ORDER")
    print("  Order: WPColl% < secondary_coll < DetRate(scen) > FA(scen) < Sev")
    print('=' * 70)

    # Compute composite score: lower WPColl% > lower 2nd > higher Det(s) > lower FA(s) > lower Sev
    # We'll lexicographically sort: (-wpc rank, -sec rank, +det_s rank, -fa_s rank, -sev rank)
    # Where higher rank = better
    def rank_metric(items, key, ascending=True):
        sorted_items = sorted(items, key=key, reverse=not ascending)
        rank_map = {}
        for i, item in enumerate(sorted_items):
            rank_map[item['name']] = i  # 0 = best
        return rank_map

    rank_wpc = rank_metric(summary, lambda x: x['wpc'], ascending=True)  # lower better
    rank_sec = rank_metric(summary, lambda x: x['sec'], ascending=True)
    rank_det = rank_metric(summary, lambda x: -x['det_s'], ascending=True)  # higher better
    rank_fa = rank_metric(summary, lambda x: x['fa_s'], ascending=True)
    rank_sev = rank_metric(summary, lambda x: x['sev'], ascending=True)
    # Composite: weighted sum (lower better)
    weights = [10, 5, 3, 2, 1]  # WPColl most, severity least
    for s in summary:
        s['score'] = (rank_wpc[s['name']] * weights[0] +
                      rank_sec[s['name']] * weights[1] +
                      rank_det[s['name']] * weights[2] +
                      rank_fa[s['name']] * weights[3] +
                      rank_sev[s['name']] * weights[4])
    summary.sort(key=lambda x: x['score'])
    print(f"  {'Rank':>4s}  {'Method':25s} {'Score':>6s} {'WPC%':>6s} {'2nd':>4s} "
          f"{'Det(s)':>7s} {'FA(s)':>6s} {'Sev':>5s}")
    print('  ' + '-' * 70)
    for i, s in enumerate(summary):
        print(f"  {i+1:>4d}  {s['name']:25s} {s['score']:5d} "
              f"{s['wpc']:5.1%} {s['sec']:4d} {s['det_s']:6.0%} "
              f"{s['fa_s']:5.0%} {s['sev']:5.2f}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

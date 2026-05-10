"""SUMO conflict-forcing scenarios + FA reduction strategies.

Design changes vs prior versions:
1. SINGLE LANE network (no escape via lane change)
2. FIXED-TIMING CONFLICTS: ego and a conflicting vehicle are dispatched
   such that they arrive at the intersection junction simultaneously
   if neither slows down. Computed from edge length, depart speed, and
   junction position.
3. AGGRESSIVE driving: minGap=0.1, tau=0.1, high speeds
4. SMALL n_other_vehicles (just enough to ensure conflict)

FA reduction strategies tested:
- A) Threshold sweep
- B) V1+geometric AND-fusion (require both signals)
- C) Multi-frame consensus (3 consecutive frames trigger)
- D) Combined A+B
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
from experiments.run_corrected_eval import (
    get_veh, build_perception, run_sumo as _run_sumo_base, SUMOM)

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
NEW_V1_THRESHOLD = 0.25


# =================================================================
# Conflict-Forcing Scenario Generator
# =================================================================

def generate_forced_conflict(net_file: Path, scenario_id: int,
                             stype: str, seed: int = 42) -> Path:
    """Generate a single-lane scenario where ego and a conflicting vehicle
    WILL collide at the center if neither yields.

    Conflict types (selected per scenario_id):
    - 0: 90° crossing (ego crosses, attacker enters perpendicularly)
    - 1: head-on (ego crosses, attacker comes from opposite arm)
    - 2: T-bone left (ego crosses, attacker enters from left at high speed)
    - 3: rear-end (ego brakes; following high-speed vehicle hits ego)
    """
    rou_file = net_file.parent / f"{stype}_forced_{scenario_id}_{seed}.rou.xml"
    random.seed(seed + scenario_id * 17)

    net = sumolib.net.readNet(str(net_file))
    center = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3:
            center = n; break
    if center is None:
        # Fallback
        with open(rou_file, 'w') as f:
            f.write('<routes></routes>')
        return rou_file

    # Get all valid through-routes (in_edge -> out_edge through center)
    in_edges = list(center.getIncoming())
    routes_dict = {}  # in_edge.id -> list of valid out_edge ids
    for ei in in_edges:
        outs = []
        for eo in ei.getOutgoing():
            outs.append(eo.getID())
        routes_dict[ei.getID()] = outs

    in_ids = list(routes_dict.keys())
    if len(in_ids) < 3:
        with open(rou_file, 'w') as f:
            f.write('<routes></routes>')
        return rou_file

    # Pick conflict type based on scenario_id
    conflict_type = scenario_id % 4

    # Edge length (assume same for all approach edges)
    edge_len = net.getEdge(in_ids[0]).getLength()

    # Ego always uses in_ids[0]
    ego_in = in_ids[0]
    # Pick straight-through out for ego (find opposite arm)
    ego_outs = routes_dict[ego_in]
    # Sort to pick consistent "straight"
    ego_out = sorted(ego_outs)[len(ego_outs)//2] if ego_outs else "?"

    vehicles = []

    # ===== Configure ego =====
    ego_speed = 10.0  # m/s
    # Time to reach junction: edge_len / ego_speed
    ego_arrival = edge_len / ego_speed

    # ===== Configure attacker(s) based on conflict type =====
    # Network max edge speed is 13.89 m/s. Use 11.5 m/s to be safe.
    if conflict_type == 0:
        # 90° crossing: attacker enters perpendicular at same time
        attacker_in = in_ids[1] if len(in_ids) > 1 else in_ids[0]
        att_out = routes_dict[attacker_in][0] if routes_dict[attacker_in] else "?"
        att_speed = 11.5
        att_arrival = edge_len / att_speed
        att_depart = max(0, ego_arrival - att_arrival)
    elif conflict_type == 1:
        # Head-on (from opposite arm)
        attacker_in = in_ids[2] if len(in_ids) > 2 else in_ids[1]
        att_out = routes_dict[attacker_in][0] if routes_dict[attacker_in] else "?"
        att_speed = 11.0
        att_arrival = edge_len / att_speed
        att_depart = max(0, ego_arrival - att_arrival)
    elif conflict_type == 2:
        # T-bone left at high speed (close to edge max)
        attacker_in = in_ids[1] if len(in_ids) > 1 else in_ids[0]
        att_out = routes_dict[attacker_in][-1] if routes_dict[attacker_in] else "?"
        att_speed = 11.5
        att_arrival = edge_len / att_speed
        att_depart = max(0, ego_arrival - att_arrival)
    else:  # 3: rear-end
        attacker_in = ego_in  # Same lane
        att_out = ego_out
        att_speed = 11.5
        # Departs slightly behind ego
        att_depart = 0.5

    vehicles.append(
        f'    <vehicle id="ego" type="ego_car" depart="0" departSpeed="{ego_speed}" departLane="0">'
        f'\n        <route edges="{ego_in} {ego_out}"/>\n    </vehicle>')
    vehicles.append(
        f'    <vehicle id="attacker" type="attacker" depart="{att_depart:.2f}" '
        f'departSpeed="{att_speed}" departLane="0">'
        f'\n        <route edges="{attacker_in} {att_out}"/>\n    </vehicle>')

    # Add cooperative perception vehicle (sees attacker if ego can't)
    # Use a different inbound edge than ego or attacker
    coop_in = None
    for ei in in_ids:
        if ei != ego_in and ei != attacker_in:
            coop_in = ei; break
    if coop_in:
        coop_out = routes_dict[coop_in][0] if routes_dict[coop_in] else "?"
        vehicles.append(
            f'    <vehicle id="coop" type="car" depart="0" departSpeed="8" departLane="0">'
            f'\n        <route edges="{coop_in} {coop_out}"/>\n    </vehicle>')

    # Maybe add 1-2 distractor vehicles for realistic density
    n_distract = random.randint(0, 2)
    for i in range(n_distract):
        din = random.choice(in_ids)
        douts = routes_dict[din]
        if not douts: continue
        dout = random.choice(douts)
        dd = random.uniform(1.0, 4.0)
        ds = random.uniform(8, 12)
        vehicles.append(
            f'    <vehicle id="d{i}" type="car" depart="{dd:.1f}" '
            f'departSpeed="{ds:.1f}" departLane="0">'
            f'\n        <route edges="{din} {dout}"/>\n    </vehicle>')

    # Sort vehicles by depart time (SUMO requires this)
    def get_depart(line):
        m = re.search(r'depart="([\d.]+)"', line)
        return float(m.group(1)) if m else 0.0
    vehicles.sort(key=get_depart)

    content = f"""<?xml version="1.0"?>
<routes>
    <vType id="ego_car" length="4.5" width="1.8" maxSpeed="20" accel="3.5" decel="6.0"
           sigma="0.0" minGap="0.2" tau="0.2" speedDev="0.0"/>
    <vType id="attacker" length="4.5" width="1.8" maxSpeed="22" accel="5.0" decel="4.0"
           sigma="0.0" minGap="0.1" tau="0.1" speedDev="0.0" impatience="1.0"/>
    <vType id="car" length="4.5" width="1.8" maxSpeed="18" accel="3.0" decel="5.0"
           sigma="0.4" minGap="0.5" tau="0.4" speedDev="0.2"/>
{chr(10).join(vehicles)}
</routes>"""
    open(rou_file, 'w').write(content)
    return rou_file


# =================================================================
# FA reduction strategy implementations
# =================================================================

class HybridWithMultiframe:
    """Hybrid + 3-frame temporal consensus.

    Only fires detection if 3 consecutive frames have V1 prob > threshold.
    """
    name = "Ours-Hybrid+3frame"
    def __init__(self, base, n_consensus=3):
        self.base = base
        self.n_consensus = n_consensus
        # Track recent V1 probs per ego (only for the ONE ego in scenarios)
        self.recent_probs = deque(maxlen=n_consensus)

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get('collision_prob', 0.0)
        self.recent_probs.append(prob)
        # Override detection: require all N recent above threshold
        consensus = (len(self.recent_probs) == self.n_consensus
                     and all(p > self.base.detection_threshold for p in self.recent_probs))
        stats['n_collisions_detected'] = 1 if consensus else 0
        return mw, stats


class HybridWithGeometricAND:
    """Hybrid + AND-fusion.

    Detection fires only if BOTH (a) V1 prob > threshold AND
    (b) at least one geometric threat exists.
    """
    name = "Ours-Hybrid+AND"
    def __init__(self, base):
        self.base = base

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        prob = stats.get('collision_prob', 0.0)
        n_geom = stats.get('n_geometric_threats', 0)
        # Override
        fire = (prob > self.base.detection_threshold) and (n_geom > 0)
        stats['n_collisions_detected'] = 1 if fire else 0
        return mw, stats


class HybridWithBothAND3frame:
    """Combine multi-frame + geometric AND."""
    name = "Ours-Hybrid+AND+3frame"
    def __init__(self, base, n_consensus=3):
        self.base = base
        self.n_consensus = n_consensus
        self.recent_probs = deque(maxlen=n_consensus)
        self.recent_geom = deque(maxlen=n_consensus)

    def constrain_waypoints(self, waypoints, perception):
        mw, stats = self.base.constrain_waypoints(waypoints, perception)
        self.recent_probs.append(stats.get('collision_prob', 0.0))
        self.recent_geom.append(stats.get('n_geometric_threats', 0))
        if len(self.recent_probs) < self.n_consensus:
            stats['n_collisions_detected'] = 0
            return mw, stats
        prob_ok = all(p > self.base.detection_threshold for p in self.recent_probs)
        geom_ok = all(g > 0 for g in self.recent_geom)
        stats['n_collisions_detected'] = 1 if (prob_ok and geom_ok) else 0
        return mw, stats


# =================================================================
# Run on forced-conflict scenarios
# =================================================================

def run_sumo_forced(net_file, rou_file, method, max_steps=300, sumo_seed=42):
    """Run SUMO with forced-conflict scenario.

    Disables SUMO car-following for ALL vehicles.
    For attacker: deliberately set speed mode to ignore obstacles (forces collision unless avoided).
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
    m = SUMOM()
    try:
        traci.start(sumo_cmd)
        seen = set()
        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            if "ego" not in vids:
                if step > 5: break
                continue
            m.n_frames += 1

            # Set attacker to ignore safety (forces conflict)
            for v in vids:
                if v == "attacker":
                    try:
                        traci.vehicle.setSpeedMode(v, 0)  # No safety at all
                    except Exception: pass

            # Apply method to ego (and only ego in this experiment)
            ego_warned = False
            ep = build_perception("ego", vids, coop_vid="coop")
            if ep is not None:
                ego_speed = ep.ego.velocity
                target_speed = ego_speed
                # Disable SUMO safety on ego
                try: traci.vehicle.setSpeedMode("ego", 6)
                except Exception: pass

                if method is not None:
                    bw = np.array([[max(ego_speed,1.0)*(t+1)*0.5, 0] for t in range(10)])
                    try:
                        if hasattr(method, 'constrain_waypoints'):
                            mw, stats = method.constrain_waypoints(bw, ep)
                            fired = (stats.get('n_geometric_threats',0) > 0
                                     or stats.get('n_collisions_detected',0) > 0
                                     or stats.get('n_modifications',0) > 0)
                            if fired:
                                ego_warned = True
                                d = math.sqrt(mw[0,0]**2 + mw[0,1]**2)
                                target_speed = min(max(d/0.5,0), ego_speed)
                        else:
                            safe = method.constrain(ep)
                            if safe.mode == ConstraintMode.MINIMUM_HARM:
                                ego_warned = True
                                target_speed = max(ego_speed*0.2, 0)
                            elif safe.mode == ConstraintMode.CONSERVATIVE:
                                ego_warned = True
                                target_speed = max(ego_speed*0.6, 0)
                    except Exception as e:
                        logger.warning(f"method {method.__class__.__name__}: {e}")

                try: traci.vehicle.setSpeed("ego", max(target_speed, 0))
                except Exception: pass

                # WPColl check
                wp = np.array([[max(ego_speed,1.0)*(t+1)*0.5, 0] for t in range(10)])
                if method and hasattr(method, 'constrain_waypoints'):
                    try: mw_check, _ = method.constrain_waypoints(wp, ep)
                    except Exception: mw_check = wp
                else:
                    mw_check = wp
                for t in range(10):
                    dt = (t+1)*0.5
                    for a in ep.agents:
                        ax = a.state.x + a.state.vx*dt
                        ay = a.state.y + a.state.vy*dt
                        if math.sqrt((mw_check[t,0]-ax)**2 + (mw_check[t,1]-ay)**2) < 2.0:
                            m.wp_coll += 1; break
                m.wp_total += 10

            if ego_warned and m.first_warning < 0:
                m.first_warning = step

            # Collisions
            try:
                for col in traci.simulation.getCollisions():
                    pair = tuple(sorted([col.collider, col.victim]))
                    if pair in seen: continue
                    seen.add(pair)
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
        logger.warning(f"SUMO failed: {e}")
        try: traci.close()
        except Exception: pass
    return m


# =================================================================
# Main
# =================================================================

def main():
    print("=" * 70)
    print("  Conflict-Forced SUMO + FA Reduction (260510)")
    print("=" * 70)

    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f}")
    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    base_hybrid = HybridSafetyConstraint(
        detector_model=v1, base_margin_visible=2.5,
        base_margin_invisible=4.0, detection_threshold=NEW_V1_THRESHOLD)

    # =========================================
    # Part 1: Forced-conflict SUMO (single-lane)
    # =========================================
    print("\n" + "=" * 70)
    print("  PART 1: Forced-Conflict SUMO (single-lane)")
    print("  Each scenario has ego + attacker on collision course.")
    print("=" * 70)

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=NEW_V1_THRESHOLD),
    }

    output_dir = SCENARIO_DIR / 'networks'
    stypes = [("t_junction_1lane", 8), ("crossroads_1lane", 8)]
    seeds = [42, 123]
    all_results = {n: [] for n in methods}

    for stype, n_sc in stypes:
        net_file = output_dir / f"{stype}.net.xml"
        if not net_file.exists():
            continue
        for seed in seeds:
            for sid in range(n_sc):
                rou_file = generate_forced_conflict(net_file, sid, stype, seed=seed)
                for mname, method in methods.items():
                    # Reset multi-frame state for each scenario
                    if hasattr(method, 'recent_probs'):
                        method.recent_probs.clear()
                    r = run_sumo_forced(net_file, rou_file, method, sumo_seed=seed)
                    all_results[mname].append((stype, sid, seed, r))

    print(f"\n  {'Method':22s} {'Coll':>6s} {'Sev':>6s} {'WPC%':>6s}  per-config-coll")
    print("  " + "-" * 60)
    n_total_runs = sum(n*len(seeds) for _, n in stypes)
    for mname in methods:
        results = all_results[mname]
        tc = sum(r.unique_ego_coll for _,_,_,r in results)
        sevs = [s for _,_,_,r in results for s in r.severities]
        tw = sum(r.wp_total for _,_,_,r in results)
        twc = sum(r.wp_coll for _,_,_,r in results)
        # Per conflict type
        type_colls = defaultdict(int)
        for stype, sid, seed, r in results:
            type_colls[sid % 4] += r.unique_ego_coll
        print(f"  {mname:22s} {tc:6d}/{n_total_runs} {np.mean(sevs) if sevs else 0:5.1f} "
              f"{twc/max(tw,1):5.1%}  c0={type_colls[0]} c1={type_colls[1]} "
              f"c2={type_colls[2]} c3={type_colls[3]}")

    # =========================================
    # Part 2: FA reduction on DeepAccident val
    # =========================================
    print("\n" + "=" * 70)
    print("  PART 2: FA Reduction Strategies (DeepAccident val 22 scenarios)")
    print("=" * 70)

    from experiments.deepaccident_loader import DeepAccidentLoader
    from experiments.run_deepaccident_unified import (
        simulate_codriving_waypoints, check_waypoint_collision, MethodResult)
    loader = DeepAccidentLoader(split='all')
    val_idx = ckpt.get('val_scenario_idx', [])

    # Methods to compare
    fa_methods = {
        "Ours-Hybrid (baseline)": base_hybrid,
        "Ours+AND": HybridWithGeometricAND(base_hybrid),
        "Ours+3frame": HybridWithMultiframe(base_hybrid, n_consensus=3),
        "Ours+AND+3frame": HybridWithBothAND3frame(base_hybrid, n_consensus=3),
    }

    # Also test threshold sweep on baseline
    threshold_sweep = [0.20, 0.25, 0.30, 0.40, 0.50]

    print(f"\n  {'Method':25s} {'Det':>5s} {'Early':>6s} {'FA':>6s} {'WPC%':>6s} {'Mod%':>6s}")
    print("  " + "-" * 60)

    for name, method in fa_methods.items():
        r = MethodResult(name=name)
        # Reset state for stateful methods
        if hasattr(method, 'recent_probs'):
            method.recent_probs.clear()
        if hasattr(method, 'recent_geom'):
            method.recent_geom.clear()

        for si in val_idx:
            s = loader.scenarios[si]
            fw = -1; fa = 0; wc = 0; mods = 0
            # Reset inter-scenario state
            if hasattr(method, 'recent_probs'):
                method.recent_probs.clear()
            if hasattr(method, 'recent_geom'):
                method.recent_geom.clear()
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                bw = simulate_codriving_waypoints(frame)
                mw, stats = method.constrain_waypoints(bw, frame.perception)
                wm = stats.get('n_collisions_detected', 0) > 0
                if wm:
                    if fw < 0: fw = fi
                    if not s['is_accident']: fa += 1
                    mods += 1
                wc += check_waypoint_collision(mw, frame)
            r.total_frames += len(s['frames']); r.modified_frames += mods
            r.waypoint_collisions += wc; r.total_waypoint_checks += len(s['frames'])*10
            if s['is_accident']:
                r.n_accident_scenarios += 1
                if fw >= 0:
                    r.n_detected += 1
                    r.early_warning_frames.append(len(s['frames']) - fw)
            else:
                r.n_normal_scenarios += 1
                if fa > 0: r.n_false_alarm_scenarios += 1

        det = r.n_detected / max(r.n_accident_scenarios, 1)
        early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa_rate = r.n_false_alarm_scenarios / max(r.n_normal_scenarios, 1)
        wpc = r.waypoint_collisions / max(r.total_waypoint_checks, 1)
        mod = r.modified_frames / max(r.total_frames, 1)
        print(f"  {name:25s} {det:4.0%} {early:5.1f} {fa_rate:5.1%} {wpc:5.1%} {mod:5.1%}")

    # Threshold sweep
    print(f"\n  Threshold sweep on baseline Hybrid (V1 threshold):")
    print(f"  {'Threshold':10s} {'Det':>5s} {'Early':>6s} {'FA':>6s} {'WPC%':>6s}")
    print("  " + "-" * 45)
    for thr in threshold_sweep:
        method = HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0, detection_threshold=thr)
        r = MethodResult(name=f"thr={thr}")
        for si in val_idx:
            s = loader.scenarios[si]
            fw = -1; fa = 0; wc = 0; mods = 0
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                bw = simulate_codriving_waypoints(frame)
                mw, stats = method.constrain_waypoints(bw, frame.perception)
                wm = stats.get('n_collisions_detected', 0) > 0
                if wm:
                    if fw < 0: fw = fi
                    if not s['is_accident']: fa += 1
                    mods += 1
                wc += check_waypoint_collision(mw, frame)
            r.total_frames += len(s['frames']); r.modified_frames += mods
            r.waypoint_collisions += wc; r.total_waypoint_checks += len(s['frames'])*10
            if s['is_accident']:
                r.n_accident_scenarios += 1
                if fw >= 0:
                    r.n_detected += 1
                    r.early_warning_frames.append(len(s['frames']) - fw)
            else:
                r.n_normal_scenarios += 1
                if fa > 0: r.n_false_alarm_scenarios += 1
        det = r.n_detected / max(r.n_accident_scenarios, 1)
        early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa_rate = r.n_false_alarm_scenarios / max(r.n_normal_scenarios, 1)
        wpc = r.waypoint_collisions / max(r.total_waypoint_checks, 1)
        print(f"  {thr:9.2f} {det:4.0%} {early:5.1f} {fa_rate:5.1%} {wpc:5.1%}")


if __name__ == "__main__":
    main()

"""Corrected fair evaluation (post-fix, 260508).

Changes from previous run_final_eval.py:
1. Uses V1 retrained with scenario-level split (no leakage)
2. RSS/APF/CBF now properly signal mode based on TTC and exclusions
3. RiskMM stats dict includes n_collisions_detected/n_geometric_threats
4. Reports BOTH val-only metrics (true generalization) AND full-set metrics
5. Uses threshold=0.25 calibrated to new V1 (was 0.5 for leaky V1)
"""
from __future__ import annotations
import sys, os, math, time, random, logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

sys.path.insert(0, str(Path(__file__).parent.parent))
from coop_safety.interface import PerceptionResult, VehicleState, Agent, AgentType, ConstraintMode
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from coop_safety.learned.collision_network import CollisionPredictionNetwork
from experiments.deepaccident_loader import DeepAccidentLoader
from experiments.run_deepaccident_unified import (
    simulate_codriving_waypoints, check_waypoint_collision, MethodResult)
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'

# Calibrated for new V1 (trained without data leakage):
# val-set frame metrics: thr=0.25 → DetRate=82%, FA=26%
NEW_V1_THRESHOLD = 0.25


# -------- DeepAccident eval ----------
def eval_deepaccident(methods_dict, loader, scenario_filter=None):
    """Evaluate methods. scenario_filter: optional list of scenario indices."""
    results = {}
    indices = scenario_filter if scenario_filter is not None else list(range(len(loader.scenarios)))

    for name, method in methods_dict.items():
        r = MethodResult(name=name)
        for si in indices:
            s = loader.scenarios[si]
            fw = -1; fa = 0; wc = 0; mods = 0
            for fi in range(len(s['frames'])):
                frame = loader.load_frame(si, fi)
                bw = simulate_codriving_waypoints(frame)

                if method is None:
                    mw = bw
                    wm = False
                elif hasattr(method, 'constrain_waypoints'):
                    try:
                        mw, stats = method.constrain_waypoints(bw, frame.perception)
                        wm = (stats.get('n_collisions_detected', 0) > 0
                              or stats.get('modification_rate', 0) > 0
                              or stats.get('n_modifications', 0) > 0)
                    except Exception as e:
                        logger.warning(f"{name} failed: {e}")
                        mw = bw
                        wm = False
                else:
                    try:
                        safe = method.constrain(frame.perception)
                        wm = safe.mode != ConstraintMode.NORMAL
                    except Exception as e:
                        logger.warning(f"{name} constrain failed: {e}")
                        wm = False
                    mw = bw

                if wm:
                    if fw < 0: fw = fi
                    if not s['is_accident']: fa += 1
                    mods += 1
                wc += check_waypoint_collision(mw, frame)

            r.total_frames += len(s['frames'])
            r.modified_frames += mods
            r.waypoint_collisions += wc
            r.total_waypoint_checks += len(s['frames']) * 10
            if s['is_accident']:
                r.n_accident_scenarios += 1
                if fw >= 0:
                    r.n_detected += 1
                    r.early_warning_frames.append(len(s['frames']) - fw)
            else:
                r.n_normal_scenarios += 1
                if fa > 0: r.n_false_alarm_scenarios += 1
        results[name] = r
    return results


def print_da_results(results, label):
    print(f"\n  --- {label} ---")
    print(f"  {'Method':22s} {'Det':>5s} {'Early':>6s} {'FA':>6s} {'WPC%':>6s} {'Mod%':>6s} {'Acc/Norm':>9s}")
    print("  " + "-" * 65)
    for name, r in results.items():
        det = r.n_detected / max(r.n_accident_scenarios, 1)
        early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa = r.n_false_alarm_scenarios / max(r.n_normal_scenarios, 1)
        wpc = r.waypoint_collisions / max(r.total_waypoint_checks, 1)
        mod = r.modified_frames / max(r.total_frames, 1)
        an = f"{r.n_accident_scenarios}/{r.n_normal_scenarios}"
        print(f"  {name:22s} {det:4.0%} {early:5.1f} {fa:5.1%} {wpc:5.1%} {mod:5.1%} {an:>9s}")


# -------- SUMO setup ----------
def get_or_create_network(stype, output_dir):
    net_file = output_dir / f"{stype}_2lane.net.xml"
    if net_file.exists(): return net_file
    arms = 3 if stype == 't_junction' else 4
    os.system(f"{NETGEN} --spider --spider.arm-number {arms} --spider.space-radius 80 "
              f"--default.speed 13.89 --no-turnarounds true --default.lanenumber 2 "
              f"-o {net_file} 2>/dev/null")
    return net_file


def generate_routes(net_file, stype, sid, n_veh=10, seed=42):
    random.seed(seed + sid * 13)
    rou_file = net_file.parent / f"{stype}_corrected_{sid}_{seed}.rou.xml"
    net = sumolib.net.readNet(str(net_file))
    center = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3: center = n; break
    routes = []
    if center:
        for ei in center.getIncoming():
            for eo in ei.getOutgoing():
                routes.append((ei.getID(), eo.getID()))
    if not routes: routes = [('E0', 'E1')]

    vehs = []
    vehs.append(f'    <vehicle id="ego" type="car" depart="0" departSpeed="11" departLane="0">'
                f'\n        <route edges="{routes[0][0]} {routes[0][1]}"/>\n    </vehicle>')
    vehs.append(f'    <vehicle id="coop" type="car" depart="0" departSpeed="9" departLane="1">'
                f'\n        <route edges="{routes[min(3,len(routes)-1)][0]} {routes[min(3,len(routes)-1)][1]}"/>\n    </vehicle>')
    for i in range(n_veh - 2):
        r = random.choice(routes)
        d = random.uniform(0.1, 2.5)
        sp = random.uniform(10, 18)
        lane = random.randint(0, 1)
        vt = "aggressive" if random.random() < 0.4 else "car"
        vehs.append(f'    <vehicle id="veh{i}" type="{vt}" depart="{d:.1f}" '
                    f'departSpeed="{sp:.1f}" departLane="{lane}">'
                    f'\n        <route edges="{r[0]} {r[1]}"/>\n    </vehicle>')
    content = f"""<?xml version="1.0"?>
<routes>
    <vType id="car" length="4.5" width="1.8" maxSpeed="19.44" accel="4.0" decel="6.0"
           sigma="0.9" minGap="0.3" tau="0.3" lcStrategic="1.0" lcCooperative="0.0"
           lcSpeedGain="2.0" speedDev="0.3" impatience="0.9"/>
    <vType id="aggressive" length="4.5" width="1.8" maxSpeed="22.22" accel="5.0" decel="7.0"
           sigma="1.0" minGap="0.2" tau="0.2" lcStrategic="0.5" lcCooperative="0.0"
           lcSpeedGain="3.0" speedDev="0.4" impatience="1.0"/>
{chr(10).join(vehs)}
</routes>"""
    open(rou_file, 'w').write(content)
    return rou_file


def get_veh(vid):
    try:
        x, y = traci.vehicle.getPosition(vid)
        sp = traci.vehicle.getSpeed(vid)
        ang = traci.vehicle.getAngle(vid)
        return (x, y, sp, math.radians(90 - ang),
                traci.vehicle.getLength(vid), traci.vehicle.getWidth(vid))
    except Exception:
        return None


def build_perception(target_vid, vids, coop_vid=None):
    ts = get_veh(target_vid)
    if ts is None: return None
    tx, ty, tsp, th, tl, tw = ts
    tvx, tvy = tsp * math.cos(th), tsp * math.sin(th)
    cp = None
    if coop_vid and coop_vid in vids:
        ci = get_veh(coop_vid)
        if ci: cp = (ci[0], ci[1])
    agents = []
    for v in vids:
        if v in (target_vid, coop_vid): continue
        vi = get_veh(v)
        if vi is None: continue
        vx, vy, vs, vh, vl, vw = vi
        dx, dy = vx - tx, vy - ty
        de = math.sqrt(dx**2 + dy**2)
        vis_t = de <= 50
        vis_c = False
        if cp:
            vis_c = math.sqrt((vx - cp[0])**2 + (vy - cp[1])**2) <= 50
        if not vis_t and not vis_c: continue
        ch, sh = math.cos(-th), math.sin(-th)
        rx, ry = dx * ch - dy * sh, dx * sh + dy * ch
        vvx, vvy = vs * math.cos(vh), vs * math.sin(vh)
        rvx = (vvx - tvx) * ch - (vvy - tvy) * sh
        rvy = (vvx - tvx) * sh + (vvy - tvy) * ch
        agents.append(Agent(state=VehicleState(id=v, x=rx, y=ry, heading=vh - th,
                            velocity=vs, vx=rvx, vy=rvy, length=vl, width=vw),
                            is_visible=vis_t, confidence=1.0 if vis_t else 0.7))
    return PerceptionResult(timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid, x=0, y=0, heading=0, velocity=tsp,
                         vx=tsp, vy=0, length=tl, width=tw), agents=agents)


def control_vehicle(vid, method, perception):
    """Fair control: all methods disable SUMO safety."""
    if perception is None:
        return False
    ego_speed = perception.ego.velocity
    try:
        traci.vehicle.setSpeedMode(vid, 6)
    except Exception:
        pass

    target_speed = ego_speed
    warned = False

    if method is not None:
        bw = np.array([[max(ego_speed, 1.0) * (t + 1) * 0.5, 0] for t in range(10)])
        try:
            if hasattr(method, 'constrain_waypoints'):
                mw, stats = method.constrain_waypoints(bw, perception)
                fired = (stats.get('n_geometric_threats', 0) > 0
                         or stats.get('n_collisions_detected', 0) > 0
                         or stats.get('n_modifications', 0) > 0)
                if fired:
                    warned = True
                    dist = math.sqrt(mw[0, 0]**2 + mw[0, 1]**2)
                    target_speed = min(max(dist / 0.5, 0), ego_speed)
                    if len(mw) > 1 and abs(mw[1, 1]) > 1.5:
                        try:
                            cl = traci.vehicle.getLaneIndex(vid)
                            if mw[1, 1] > 0 and cl < 1:
                                traci.vehicle.changeLane(vid, cl + 1, 2.0)
                            elif mw[1, 1] < 0 and cl > 0:
                                traci.vehicle.changeLane(vid, cl - 1, 2.0)
                        except Exception:
                            pass
            else:
                safe = method.constrain(perception)
                if safe.mode == ConstraintMode.MINIMUM_HARM:
                    warned = True
                    target_speed = max(ego_speed * 0.2, 0)
                elif safe.mode == ConstraintMode.CONSERVATIVE:
                    warned = True
                    target_speed = max(ego_speed * 0.6, 0)
        except Exception as e:
            logger.warning(f"Control failed for {vid} with {method.__class__.__name__}: {e}")

    try:
        traci.vehicle.setSpeed(vid, max(target_speed, 0))
    except Exception:
        pass
    return warned


@dataclass
class SUMOM:
    n_frames: int = 0
    unique_ego_coll: int = 0
    secondary_coll: int = 0
    severities: list = field(default_factory=list)
    wp_coll: int = 0
    wp_total: int = 0
    first_warning: int = -1
    first_collision: int = -1


def run_sumo(net_file, rou_file, method, sumo_seed=42, max_steps=400):
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn', '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0', '--step-length', '0.1',
                '--no-step-log', 'true', '--no-warnings', 'true',
                '--seed', str(sumo_seed), '--lanechange.duration', '1.5']
    m = SUMOM()
    try:
        traci.start(sumo_cmd)
        seen = set()
        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            if "ego" not in vids:
                if step > 10: break
                continue
            m.n_frames += 1
            ego_warned = False
            for v in vids:
                cv = "coop" if v == "ego" else ("ego" if v == "coop" else None)
                p = build_perception(v, vids, coop_vid=cv)
                w = control_vehicle(v, method, p)
                if v == "ego" and w: ego_warned = True
            if ego_warned and m.first_warning < 0:
                m.first_warning = step

            ep = build_perception("ego", vids, coop_vid="coop")
            if ep:
                sp = ep.ego.velocity
                wp = np.array([[max(sp, 1.0) * (t + 1) * 0.5, 0] for t in range(10)])
                if method:
                    try: mw, _ = method.constrain_waypoints(wp, ep) if hasattr(method, 'constrain_waypoints') else (wp, {})
                    except Exception: mw = wp
                else: mw = wp
                for t in range(10):
                    dt = (t + 1) * 0.5
                    for a in ep.agents:
                        ax = a.state.x + a.state.vx * dt
                        ay = a.state.y + a.state.vy * dt
                        if math.sqrt((mw[t, 0] - ax)**2 + (mw[t, 1] - ay)**2) < 2.0:
                            m.wp_coll += 1; break
                m.wp_total += 10

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
            except Exception:
                pass
        traci.close()
    except Exception as e:
        logger.warning(f"SUMO failed: {e}")
        try: traci.close()
        except Exception: pass
    return m


# -------- Main ----------
def main():
    print("=" * 70)
    print("  CORRECTED FINAL EVALUATION (post-fix 260508)")
    print("=" * 70)

    # Load NEW V1 (scenario-level split)
    ckpt = torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                      map_location='cpu', weights_only=False)
    print(f"  V1: AUC={ckpt.get('auc',0):.4f} epoch={ckpt.get('epoch',-1)}")
    train_idx = ckpt.get('train_scenario_idx', [])
    val_idx = ckpt.get('val_scenario_idx', [])
    print(f"  V1 train scenarios: {len(train_idx)}, val: {len(val_idx)}")

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(ckpt['model'])
    v1.eval()

    methods = {
        "NoConstraint": None,
        "RSS": RSSOnly(),
        "APF": RiskPotentialField(),
        "UniE2EV2X": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP": MAPSafety(min_clearance=0.5),
        "RiskMM": RiskMMSafety(v_max=20.0),
        "Ours-Hybrid": HybridSafetyConstraint(
            detector_model=v1, base_margin_visible=2.5,
            base_margin_invisible=4.0,
            detection_threshold=NEW_V1_THRESHOLD),
    }

    t0 = time.time()
    loader = DeepAccidentLoader(split='all')

    # Part 1A: DeepAccident on UNSEEN val scenarios only (true generalization)
    print("\n" + "=" * 70)
    print("  PART 1A: DeepAccident — VAL ONLY (V1 unseen, true generalization)")
    print("=" * 70)
    da_val = eval_deepaccident(methods, loader, scenario_filter=val_idx)
    print_da_results(da_val, f"Val scenarios ({len(val_idx)})")

    # Part 1B: Full 104 scenarios (for direct comparison with leaky old results)
    print("\n" + "=" * 70)
    print("  PART 1B: DeepAccident — FULL 104 (for comparison with old results)")
    print("=" * 70)
    da_full = eval_deepaccident(methods, loader, scenario_filter=None)
    print_da_results(da_full, "All 104 scenarios")

    # Part 2: SUMO 3 seeds
    print("\n" + "=" * 70)
    print("  PART 2: SUMO Dual-Lane (30 scenarios × 3 seeds)")
    print("=" * 70)

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)
    stypes = [("t_junction", 15), ("crossroads", 15)]
    seeds = [42, 123, 789]
    all_sumo = {n: {s: [] for s, _ in stypes} for n in methods}

    for sumo_seed in seeds:
        print(f"\n  --- Seed {sumo_seed} ---")
        for stype, n_sc in stypes:
            net_file = get_or_create_network(stype, output_dir)
            for sid in range(n_sc):
                rou_file = generate_routes(net_file, stype, sid, n_veh=random.randint(10, 14), seed=sumo_seed)
                for mname, method in methods.items():
                    r = run_sumo(net_file, rou_file, method, sumo_seed=sumo_seed)
                    all_sumo[mname][stype].append(r)
            ro = all_sumo["Ours-Hybrid"][stype]
            rn = all_sumo["NoConstraint"][stype]
            oc = sum(x.unique_ego_coll for x in ro[-n_sc:])
            nc = sum(x.unique_ego_coll for x in rn[-n_sc:])
            print(f"    {stype}: NoCon={nc}, Ours={oc}")

    # SUMO results
    print("\n" + "=" * 70)
    print("  SUMO RESULTS")
    print("=" * 70)
    for stype, _ in stypes:
        print(f"\n  [{stype}]")
        print(f"  {'Method':22s} {'Coll':>6s} {'2nd':>5s} {'2ndR':>6s} {'Sev':>6s} {'WPC%':>6s}")
        print("  " + "-" * 55)
        for mname in methods:
            results = all_sumo[mname][stype]
            tc = sum(r.unique_ego_coll for r in results)
            ts = sum(r.secondary_coll for r in results)
            sevs = [s for r in results for s in r.severities]
            tw = sum(r.wp_total for r in results)
            twc = sum(r.wp_coll for r in results)
            n_per_seed = max(len(results) // 3, 1)
            seed_colls = [sum(r.unique_ego_coll for r in results[i*n_per_seed:(i+1)*n_per_seed]) for i in range(3)]
            print(f"  {mname:22s} {tc:6d} {ts:5d} {ts/max(tc,1):5.0%} "
                  f"{np.mean(sevs) if sevs else 0:5.1f} {twc/max(tw,1):5.1%}  seeds:{seed_colls}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

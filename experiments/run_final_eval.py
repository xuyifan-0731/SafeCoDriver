"""FINAL VERSION: Fair evaluation on SUMO + DeepAccident.

Fairness fixes (260508):
1. ALL methods (including NoConstraint) disable SUMO car-following
   NoConstraint = CoDriving only (constant velocity, no safety logic)
2. Collision deduplication: same pair counted once
3. Multiple seeds for variance estimation
4. Both platforms evaluated with identical algorithm
5. V1 model trained with scenario-level split (no leakage)
6. RSS/APF/CBF mode signaling fixed (was hardcoded NORMAL)
7. RiskMM stats dict now includes n_collisions_detected/n_geometric_threats

Evaluation:
- DeepAccident (104 scenarios): DetRate, EarlyWarn, FalseAlm, WPColl%, ModRate
- SUMO (30 scenarios × 3 seeds): UniqueCollisions, SecondaryColl, SecRate, Severity, WPColl%
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
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety

import torch

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
NETGEN = str(Path(sys.executable).parent / 'netgenerate')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'


# =================================================================
# SUMO Infrastructure
# =================================================================

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
    rou_file = net_file.parent / f"{stype}_final_{sid}_{seed}.rou.xml"
    net = sumolib.net.readNet(str(net_file))
    center = None
    for n in net.getNodes():
        if len(n.getIncoming()) >= 3: center = n; break
    routes = []
    if center:
        for ei in center.getIncoming():
            for eo in ei.getOutgoing():
                routes.append((ei.getID(), eo.getID()))
    if not routes: routes = [('E0','E1')]

    vehs = []
    vehs.append(f'    <vehicle id="ego" type="car" depart="0" departSpeed="11" departLane="0">'
                f'\n        <route edges="{routes[0][0]} {routes[0][1]}"/>\n    </vehicle>')
    vehs.append(f'    <vehicle id="coop" type="car" depart="0" departSpeed="9" departLane="1">'
                f'\n        <route edges="{routes[min(3,len(routes)-1)][0]} {routes[min(3,len(routes)-1)][1]}"/>\n    </vehicle>')
    for i in range(n_veh-2):
        r = random.choice(routes)
        d = random.uniform(0.1, 2.5)
        sp = random.uniform(10, 18)
        lane = random.randint(0,1)
        vtype = "aggressive" if random.random() < 0.4 else "car"
        vehs.append(f'    <vehicle id="veh{i}" type="{vtype}" depart="{d:.1f}" '
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
    open(rou_file,'w').write(content)
    return rou_file

def get_veh_info(vid):
    try:
        x,y = traci.vehicle.getPosition(vid)
        sp = traci.vehicle.getSpeed(vid)
        ang = traci.vehicle.getAngle(vid)
        h = math.radians(90-ang)
        l = traci.vehicle.getLength(vid)
        w = traci.vehicle.getWidth(vid)
        return (x,y,sp,h,l,w)
    except: return None

def build_perception(target_vid, veh_ids, coop_vid=None):
    ts = get_veh_info(target_vid)
    if ts is None: return None
    tx,ty,tsp,th,tl,tw = ts
    tvx,tvy = tsp*math.cos(th), tsp*math.sin(th)
    cp = None
    if coop_vid and coop_vid in veh_ids:
        ci = get_veh_info(coop_vid)
        if ci: cp = (ci[0],ci[1])
    agents = []
    for vid in veh_ids:
        if vid in (target_vid, coop_vid): continue
        vi = get_veh_info(vid)
        if vi is None: continue
        vx,vy,vs,vh,vl,vw = vi
        dx,dy = vx-tx, vy-ty
        de = math.sqrt(dx**2+dy**2)
        vis_t = de <= 50
        vis_c = False
        if cp: vis_c = math.sqrt((vx-cp[0])**2+(vy-cp[1])**2) <= 50
        if not vis_t and not vis_c: continue
        ch,sh = math.cos(-th),math.sin(-th)
        rx,ry = dx*ch-dy*sh, dx*sh+dy*ch
        vvx,vvy = vs*math.cos(vh), vs*math.sin(vh)
        rvx = (vvx-tvx)*ch-(vvy-tvy)*sh
        rvy = (vvx-tvx)*sh+(vvy-tvy)*ch
        agents.append(Agent(state=VehicleState(id=vid,x=rx,y=ry,heading=vh-th,
                           velocity=vs,vx=rvx,vy=rvy,length=vl,width=vw),
                           is_visible=vis_t,confidence=1.0 if vis_t else 0.7))
    return PerceptionResult(timestamp=traci.simulation.getTime(),
        ego=VehicleState(id=target_vid,x=0,y=0,heading=0,velocity=tsp,vx=tsp,vy=0,length=tl,width=tw),
        agents=agents)


def control_vehicle(vid, method, perception):
    """FAIR control: ALL methods disable SUMO safety, use CoDriving + constraint.

    NoConstraint = CoDriving only (maintain velocity, no safety logic).
    Others = CoDriving + safety constraint (may reduce speed / change lane).
    """
    if perception is None:
        return False  # No warning

    ego_speed = perception.ego.velocity

    # FAIRNESS: disable SUMO car-following for ALL methods (including NoConstraint)
    try:
        traci.vehicle.setSpeedMode(vid, 6)
    except:
        pass

    # CoDriving baseline: maintain current speed
    target_speed = ego_speed
    warned = False

    if method is not None:
        base_wp = np.array([[max(ego_speed,1.0)*(t+1)*0.5, 0] for t in range(10)])
        try:
            if hasattr(method, 'constrain_waypoints'):
                mod_wp, stats = method.constrain_waypoints(base_wp, perception)
                n_threats = stats.get('n_geometric_threats', 0)
                n_det = stats.get('n_collisions_detected', 0)
                n_mod = stats.get('n_modifications', 0)  # for RiskMM compat

                if n_threats > 0 or n_det > 0 or n_mod > 0:
                    warned = True
                    dist = math.sqrt(mod_wp[0,0]**2 + mod_wp[0,1]**2)
                    target_speed = min(max(dist/0.5, 0), ego_speed)

                    # Lateral avoidance
                    if len(mod_wp) > 1 and abs(mod_wp[1,1]) > 1.5:
                        try:
                            cl = traci.vehicle.getLaneIndex(vid)
                            if mod_wp[1,1] > 0 and cl < 1:
                                traci.vehicle.changeLane(vid, cl+1, 2.0)
                            elif mod_wp[1,1] < 0 and cl > 0:
                                traci.vehicle.changeLane(vid, cl-1, 2.0)
                        except Exception as e:
                            logger.debug(f"Lane change failed for {vid}: {e}")
            else:
                # Old-style methods (RSS, etc.)
                safe = method.constrain(perception)
                if safe.mode == ConstraintMode.MINIMUM_HARM:
                    warned = True
                    target_speed = max(ego_speed * 0.2, 0)
                elif safe.mode == ConstraintMode.CONSERVATIVE:
                    warned = True
                    target_speed = max(ego_speed * 0.6, 0)
        except Exception as e:
            logger.warning(f"Method {method.__class__.__name__} failed for {vid}: {e}")

    try:
        traci.vehicle.setSpeed(vid, max(target_speed, 0))
    except:
        pass

    return warned


@dataclass
class SUMOMetrics:
    n_frames: int = 0
    unique_ego_coll: int = 0
    secondary_coll: int = 0
    severities: list = field(default_factory=list)
    wp_coll: int = 0
    wp_total: int = 0
    first_warning: int = -1
    first_collision: int = -1
    n_warnings: int = 0


def run_sumo_scenario(net_file, rou_file, method, sumo_seed=42, max_steps=400):
    """Run one SUMO scenario with proper fairness."""
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true',
                '--seed', str(sumo_seed),
                '--lanechange.duration', '1.5']

    m = SUMOMetrics()
    try:
        traci.start(sumo_cmd)
        seen_pairs = set()

        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())
            if "ego" not in vids:
                if step > 10: break
                continue
            m.n_frames += 1

            # Control ALL vehicles (fair: same speedMode for all)
            ego_warned = False
            for vid in vids:
                coop = "coop" if vid=="ego" else ("ego" if vid=="coop" else None)
                perc = build_perception(vid, vids, coop_vid=coop)
                w = control_vehicle(vid, method, perc)
                if vid == "ego" and w:
                    ego_warned = True

            if ego_warned:
                if m.first_warning < 0: m.first_warning = step
                m.n_warnings += 1

            # Ego WPColl check
            ep = build_perception("ego", vids, coop_vid="coop")
            if ep:
                sp = ep.ego.velocity
                wp = np.array([[max(sp,1.0)*(t+1)*0.5,0] for t in range(10)])
                if method:
                    try: mw,_ = method.constrain_waypoints(wp, ep)
                    except: mw = wp
                else: mw = wp
                for t in range(10):
                    dt=(t+1)*0.5
                    for a in ep.agents:
                        ax=a.state.x+a.state.vx*dt; ay=a.state.y+a.state.vy*dt
                        if math.sqrt((mw[t,0]-ax)**2+(mw[t,1]-ay)**2)<2.0:
                            m.wp_coll+=1; break
                m.wp_total+=10

            # Collision deduplication
            try:
                for col in traci.simulation.getCollisions():
                    pair = tuple(sorted([col.collider, col.victim]))
                    if pair in seen_pairs: continue
                    seen_pairs.add(pair)
                    if "ego" not in pair: continue
                    try:
                        s1 = traci.vehicle.getSpeed(col.collider) if col.collider in vids else 0
                        s2 = traci.vehicle.getSpeed(col.victim) if col.victim in vids else 0
                        m.severities.append(abs(s1-s2))
                    except: m.severities.append(10.0)
                    m.unique_ego_coll += 1
                    if m.first_collision < 0:
                        m.first_collision = step
                    else:
                        m.secondary_coll += 1
            except: pass

        traci.close()
    except:
        try: traci.close()
        except: pass
    return m


# =================================================================
# DeepAccident Evaluation
# =================================================================

def eval_deepaccident(methods_dict, loader):
    """Full DeepAccident evaluation for all methods."""
    results = {}
    for name, method in methods_dict.items():
        r = MethodResult(name=name)
        if method is None:
            # NoConstraint: just check base waypoints
            for si, s in enumerate(loader.scenarios):
                for fi in range(len(s['frames'])):
                    frame = loader.load_frame(si, fi)
                    wp = simulate_codriving_waypoints(frame)
                    r.waypoint_collisions += check_waypoint_collision(wp, frame)
                r.total_waypoint_checks += len(s['frames'])*10
                r.total_frames += len(s['frames'])
                if s['is_accident']: r.n_accident_scenarios += 1
                else: r.n_normal_scenarios += 1
        else:
            for si, s in enumerate(loader.scenarios):
                fw=-1; fa=0; wc=0; mods=0
                for fi in range(len(s['frames'])):
                    frame = loader.load_frame(si, fi)
                    bw = simulate_codriving_waypoints(frame)

                    if hasattr(method, 'constrain_waypoints'):
                        mw, stats = method.constrain_waypoints(bw, frame.perception)
                        wm = stats.get('n_collisions_detected',0)>0 or stats.get('modification_rate',0)>0
                    else:
                        # Old-style (RSS, etc.): use constrain() interface
                        try:
                            safe = method.constrain(frame.perception)
                            wm = safe.mode != ConstraintMode.NORMAL
                        except:
                            wm = False
                        mw = bw  # Old-style doesn't modify waypoints

                    if wm:
                        if fw<0: fw=fi
                        if not s['is_accident']: fa+=1
                        mods+=1
                    wc += check_waypoint_collision(mw, frame)
                r.total_frames+=len(s['frames']); r.modified_frames+=mods
                r.waypoint_collisions+=wc; r.total_waypoint_checks+=len(s['frames'])*10
                if s['is_accident']:
                    r.n_accident_scenarios+=1
                    if fw>=0: r.n_detected+=1; r.early_warning_frames.append(len(s['frames'])-fw)
                else:
                    r.n_normal_scenarios+=1
                    if fa>0: r.n_false_alarm_scenarios+=1
        results[name] = r
    return results


# =================================================================
# Main
# =================================================================

def main():
    print("="*70)
    print("  FINAL EVALUATION: Fair Comparison on DeepAccident + SUMO")
    print("="*70)
    print("  Fairness: ALL methods disable SUMO safety, use CoDriving control")
    print("  Variance: 3 seeds for SUMO to check stability")
    print("  Dedup: same collision pair counted only once")

    v1 = CollisionPredictionNetwork()
    v1.load_state_dict(torch.load("/raid/xuyifan/jiqiuyu/models/collision_net_best.pt",
                                   map_location='cpu',weights_only=False)['model'])
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

    t0 = time.time()

    # =============================================
    # Part 1: DeepAccident (deterministic, 1 run)
    # =============================================
    print("\n" + "="*70)
    print("  PART 1: DeepAccident (104 scenarios)")
    print("="*70)

    loader = DeepAccidentLoader(split='all')
    da_results = eval_deepaccident(methods, loader)

    print(f"\n  {'Method':20s} {'Det':>5s} {'Early':>6s} {'FA':>6s} {'WPC%':>6s} {'Mod%':>6s}")
    print("  "+"-"*50)
    for name, r in da_results.items():
        det = r.n_detected/max(r.n_accident_scenarios,1)
        early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa = r.n_false_alarm_scenarios/max(r.n_normal_scenarios,1)
        wpc = r.waypoint_collisions/max(r.total_waypoint_checks,1)
        mod = r.modified_frames/max(r.total_frames,1)
        print(f"  {name:20s} {det:4.0%} {early:5.1f} {fa:5.1%} {wpc:5.1%} {mod:5.1%}")

    # =============================================
    # Part 2: SUMO (3 seeds for variance)
    # =============================================
    print("\n" + "="*70)
    print("  PART 2: SUMO Dual-Lane (30 scenarios × 3 seeds)")
    print("="*70)

    output_dir = SCENARIO_DIR / 'networks'
    output_dir.mkdir(parents=True, exist_ok=True)
    stypes = [("t_junction", 15), ("crossroads", 15)]
    seeds = [42, 123, 789]

    # Aggregate across seeds
    all_sumo = {name: {s: [] for s,_ in stypes} for name in methods}

    for seed_idx, sumo_seed in enumerate(seeds):
        print(f"\n  --- Seed {sumo_seed} (run {seed_idx+1}/3) ---")
        for stype, n_sc in stypes:
            net_file = get_or_create_network(stype, output_dir)
            for sid in range(n_sc):
                rou_file = generate_routes(net_file, stype, sid, n_veh=random.randint(10,14), seed=sumo_seed)
                for mname, method in methods.items():
                    r = run_sumo_scenario(net_file, rou_file, method, sumo_seed=sumo_seed)
                    all_sumo[mname][stype].append(r)

            # Progress
            ro = all_sumo["Ours-Hybrid"][stype]
            rn = all_sumo["NoConstraint"][stype]
            oc = sum(x.unique_ego_coll for x in ro[-n_sc:])
            nc = sum(x.unique_ego_coll for x in rn[-n_sc:])
            print(f"    {stype}: NoCon={nc} coll, Ours={oc} coll")

    # Print SUMO results
    print("\n" + "="*70)
    print("  SUMO RESULTS (averaged over 3 seeds)")
    print("="*70)

    for stype, _ in stypes:
        print(f"\n  [{stype}]")
        print(f"  {'Method':20s} {'Coll':>6s} {'2nd':>5s} {'2ndR':>6s} {'Sev':>6s} {'WPC%':>6s} {'Det':>5s}")
        print("  "+"-"*55)
        for mname in methods:
            results = all_sumo[mname][stype]
            tc=sum(r.unique_ego_coll for r in results)
            ts=sum(r.secondary_coll for r in results)
            sr=ts/max(tc,1)
            sevs=[s for r in results for s in r.severities]
            avg_sev=np.mean(sevs) if sevs else 0
            tw=sum(r.wp_total for r in results)
            twc=sum(r.wp_coll for r in results)
            wr=twc/max(tw,1)
            ncs=sum(1 for r in results if r.unique_ego_coll>0)
            nd=sum(1 for r in results if r.unique_ego_coll>0 and r.first_warning>=0
                   and (r.first_collision<0 or r.first_warning<=r.first_collision))
            dr=nd/max(ncs,1)
            # Per-seed breakdown
            n_per_seed = len(results)//3
            seed_colls = [sum(r.unique_ego_coll for r in results[i*n_per_seed:(i+1)*n_per_seed]) for i in range(3)]
            print(f"  {mname:20s} {tc:6d} {ts:5d} {sr:5.0%} {avg_sev:5.1f} {wr:5.1%} {dr:4.0%}"
                  f"  seeds:{seed_colls}")

    # Overall
    print(f"\n  {'OVERALL':20s} {'Coll':>6s} {'2nd':>5s} {'2ndR':>6s} {'Sev':>6s} {'WPC%':>6s}")
    print("  "+"-"*50)
    for mname in methods:
        tc=0;ts=0;sevs=[];tw=0;twc=0
        for stype,_ in stypes:
            for r in all_sumo[mname][stype]:
                tc+=r.unique_ego_coll; ts+=r.secondary_coll
                sevs.extend(r.severities); tw+=r.wp_total; twc+=r.wp_coll
        sr=ts/max(tc,1); avg_sev=np.mean(sevs) if sevs else 0; wr=twc/max(tw,1)
        print(f"  {mname:20s} {tc:6d} {ts:5d} {sr:5.0%} {avg_sev:5.1f} {wr:5.1%}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

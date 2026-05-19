"""Generate SUMO simulations + collision GIFs for 5 blind-spot scenarios.

Each scenario runs with NoConstraint (so the natural collision occurs).
Positions are logged every step; matplotlib renders an animation → GIF.

Output:
  paper/figures/gifs/BS{1..5}_collision.gif
  paper/figures/trajectories/BS{1..5}_trajectory.png

Networks:
  - cross_2lane: 4-way intersection, 2 lanes/direction (BS3, BS4)
  - grid_2lane: 3×3 grid, supports U-turns (BS1)
  - straight_2lane: linear road (BS2, BS5)
"""
from __future__ import annotations
import sys, os, math, logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.WARNING)
os.environ['SUMO_HOME'] = str(Path(sys.executable).parent.parent / 'lib/python3.10/site-packages/sumo')
import traci
import sumolib

SUMO_BIN = str(Path(sys.executable).parent / 'sumo')
SCENARIO_DIR = Path(__file__).parent / 'sumo_scenarios'
NET_DIR = SCENARIO_DIR / 'networks'
GIF_DIR = Path('/raid/xuyifan/jiqiuyu/paper/figures/gifs')
TRAJ_DIR = Path('/raid/xuyifan/jiqiuyu/paper/figures/trajectories')
GIF_DIR.mkdir(parents=True, exist_ok=True)
TRAJ_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# vType definitions (shared across scenarios)
# ============================================================
VTYPES = """    <vType id="ego_car" length="4.5" width="1.8" maxSpeed="13.5" accel="3.5" decel="6.0"
           sigma="0.0" minGap="0.2" tau="0.2" speedDev="0.0" color="0,0.6,0.9"/>
    <vType id="attacker" length="4.5" width="1.8" maxSpeed="14.0" accel="5.0" decel="4.0"
           sigma="0.0" minGap="0.1" tau="0.1" speedDev="0.0" impatience="1.0" color="0.9,0.2,0.2"/>
    <vType id="blocker" length="8.0" width="2.5" maxSpeed="13.5" accel="2.0" decel="6.0"
           sigma="0.0" minGap="0.5" tau="0.5" color="0.5,0.5,0.5"/>
    <vType id="pedestrian" length="0.6" width="0.6" maxSpeed="2.5" accel="2.0" decel="3.0"
           sigma="0.0" minGap="0.1" tau="0.1" color="0.9,0.2,0.2"/>
    <vType id="coop" length="4.5" width="1.8" maxSpeed="11.0" accel="3.0" decel="5.0"
           sigma="0.3" minGap="0.5" tau="0.4" color="0.15,0.7,0.4"/>
    <vType id="other" length="4.5" width="1.8" maxSpeed="11.0" accel="3.0" decel="5.0"
           sigma="0.4" minGap="0.5" tau="0.4" color="0.95,0.6,0.1"/>
    <vType id="midcar" length="4.5" width="1.8" maxSpeed="13.5" accel="3.5" decel="6.0"
           sigma="0.0" minGap="0.3" tau="0.3" color="0.95,0.6,0.1"/>"""


def write_route(out_path: Path, vehicles_xml: list) -> Path:
    import re
    def get_dep(line):
        m = re.search(r'depart="([\d.]+)"', line)
        return float(m.group(1)) if m else 0.0
    vehicles_xml.sort(key=get_dep)
    content = f"""<?xml version="1.0"?>
<routes>
{VTYPES}
{chr(10).join(vehicles_xml)}
</routes>"""
    open(out_path, 'w').write(content)
    return out_path


# ============================================================
# Scenario route generators
# ============================================================

def make_BS1(blind=True):
    """Attacker U-turns between ego and blocker, then cuts to the outer lane."""
    net = NET_DIR / 'grid_2lane.net.xml'

    # Ego and blocker are both on the eastbound inner lane. The attacker starts
    # between them on the opposite-direction inner lane, then U-turns at B1 and
    # is forced to the eastbound outer lane during the turn/merge.
    vehs = [
        f'    <vehicle id="ego" type="ego_car" depart="0" departSpeed="9.5" departLane="1" departPos="0">\n        <route edges="A1B1 B1C1"/>\n    </vehicle>',
        f'    <vehicle id="blk_bus" type="blocker" depart="0" departSpeed="5.0" departLane="1" departPos="32">\n        <route edges="A1B1 B1C1"/>\n    </vehicle>',
        f'    <vehicle id="attacker" type="attacker" depart="2.6" departSpeed="7.0" departLane="1" departPos="28">\n        <route edges="C1B1 B1C1"/>\n    </vehicle>',
    ]
    if not blind:
        vehs.append(f'    <vehicle id="coop" type="coop" depart="0" departSpeed="9" departLane="0">\n        <route edges="C1B1 B1A1"/>\n    </vehicle>')
    else:
        vehs.append(f'    <vehicle id="other" type="other" depart="0" departSpeed="9" departLane="0">\n        <route edges="C1B1 B1A1"/>\n    </vehicle>')

    rou = NET_DIR / f'BS1_uturn_{"blind" if blind else "nonblind"}.rou.xml'
    return net, write_route(rou, vehs)


def make_BS2(blind=True):
    """Pedestrian crosses in front-left of blocker while ego overtakes."""
    net = NET_DIR / 'straight_2lane.net.xml'
    route = "B0C0 C0D0"
    vehs = [
        f'    <vehicle id="ego" type="ego_car" depart="0" departSpeed="12.0" departLane="1" departPos="0">\n        <route edges="{route}"/>\n    </vehicle>',
        f'    <vehicle id="blk_park" type="blocker" depart="0" departSpeed="2.0" departLane="0" departPos="18">\n        <route edges="{route}"/>\n    </vehicle>',
        f'    <vehicle id="attacker" type="pedestrian" depart="0" departSpeed="0" departLane="0" departPos="35">\n        <route edges="{route}"/>\n    </vehicle>',
    ]
    if not blind:
        vehs.append(f'    <vehicle id="coop" type="coop" depart="0" departSpeed="9" departLane="1">\n        <route edges="C0B0 B0A0"/>\n    </vehicle>')
    else:
        vehs.append(f'    <vehicle id="other" type="other" depart="0" departSpeed="9" departLane="1">\n        <route edges="C0B0 B0A0"/>\n    </vehicle>')

    rou = NET_DIR / f'BS2_ped_{"blind" if blind else "nonblind"}.rou.xml'
    return net, write_route(rou, vehs)


def make_BS3(blind=True):
    """Corner cross: attacker straight, blocker parallel on attacker's left lane."""
    net = NET_DIR / 'cross_2lane.net.xml'
    vehs = [
        f'    <vehicle id="ego" type="ego_car" depart="0" departSpeed="12.0" departLane="0">\n        <route edges="B3A1 A1B1"/>\n    </vehicle>',
        f'    <vehicle id="attacker" type="attacker" depart="0.4" departSpeed="11.5" departLane="0">\n        <route edges="B4A1 A1B2"/>\n    </vehicle>',
        f'    <vehicle id="blk_parallel" type="blocker" depart="0.4" departSpeed="11.5" departLane="1">\n        <route edges="B4A1 A1B2"/>\n    </vehicle>',
    ]
    if not blind:
        # Coop on east arm coming westward (B1A1) — sees south arm clearly
        vehs.append(f'    <vehicle id="coop" type="coop" depart="0" departSpeed="9" departLane="0">\n        <route edges="B1A1 A1B3"/>\n    </vehicle>')
    else:
        vehs.append(f'    <vehicle id="other" type="other" depart="0" departSpeed="9" departLane="0">\n        <route edges="B1A1 A1B3"/>\n    </vehicle>')

    rou = NET_DIR / f'BS3_corner_{"blind" if blind else "nonblind"}.rou.xml'
    return net, write_route(rou, vehs)


def make_BS4(blind=True):
    """Right-turn merge: attacker right-turns from south into ego's east-bound lane.
    Blocker on ego's right lane, ahead of ego, parallel same direction.
    """
    net = NET_DIR / 'cross_2lane.net.xml'
    vehs = [
        f'    <vehicle id="ego" type="ego_car" depart="0" departSpeed="11.5" departLane="1">\n        <route edges="B3A1 A1B1"/>\n    </vehicle>',
        f'    <vehicle id="blk_right" type="blocker" depart="0" departSpeed="11.5" departLane="0">\n        <route edges="B3A1 A1B1"/>\n    </vehicle>',
        f'    <vehicle id="attacker" type="attacker" depart="0.7" departSpeed="12.0" departLane="0">\n        <route edges="B4A1 A1B1"/>\n    </vehicle>',
    ]
    if not blind:
        vehs.append(f'    <vehicle id="coop" type="coop" depart="0" departSpeed="9" departLane="0">\n        <route edges="B1A1 A1B3"/>\n    </vehicle>')
    else:
        vehs.append(f'    <vehicle id="other" type="other" depart="0" departSpeed="9" departLane="0">\n        <route edges="B1A1 A1B3"/>\n    </vehicle>')

    rou = NET_DIR / f'BS4_rightturn_{"blind" if blind else "nonblind"}.rou.xml'
    return net, write_route(rou, vehs)


def make_BS5(blind=True):
    """3-car chain on straight road. Lead brakes, mid swerves, ego must react.
    Mid-car serves as natural blocker hiding Lead from ego (BLIND mode).
    """
    net = NET_DIR / 'straight_2lane.net.xml'
    # Use 4-edge route to give longer corridor: A0B0 → B0C0 → C0D0 → D0E0 (240m total)
    # Stagger departs so all 3 fit on road: attacker first (will be at front),
    # mid next, ego last
    route = "A0B0 B0C0 C0D0 D0E0"
    vehs = [
        # Lead (attacker) - departs first → ends up furthest ahead
        f'    <vehicle id="attacker" type="attacker" depart="0" departSpeed="11" departLane="1">\n        <route edges="{route}"/>\n    </vehicle>',
        # Mid-car - departs 1.0s later
        f'    <vehicle id="midcar" type="midcar" depart="1.0" departSpeed="11" departLane="1">\n        <route edges="{route}"/>\n    </vehicle>',
        # Ego (target) - departs 2.0s later
        f'    <vehicle id="ego" type="ego_car" depart="2.0" departSpeed="11" departLane="1">\n        <route edges="{route}"/>\n    </vehicle>',
    ]
    if not blind:
        # Coop on opposite direction
        vehs.append(f'    <vehicle id="coop" type="coop" depart="0" departSpeed="9" departLane="1">\n        <route edges="E0D0 D0C0 C0B0 B0A0"/>\n    </vehicle>')
    else:
        vehs.append(f'    <vehicle id="other" type="other" depart="0" departSpeed="9" departLane="1">\n        <route edges="E0D0 D0C0 C0B0 B0A0"/>\n    </vehicle>')

    rou = NET_DIR / f'BS5_chain_{"blind" if blind else "nonblind"}.rou.xml'
    return net, write_route(rou, vehs)


# ============================================================
# Simulation runner with position logging
# ============================================================

def run_scenario_log(net_file, rou_file, scenario_name, max_steps=200):
    """Run SUMO scenario, log positions of all vehicles each step.

    Returns: list of frames, each frame = {step, time, vehicles: {id: (x, y, heading, type)}}
    """
    sumo_cmd = [SUMO_BIN, '-n', str(net_file), '-r', str(rou_file),
                '--collision.action', 'warn',
                '--collision.check-junctions', 'true',
                '--collision.mingap-factor', '0',
                '--step-length', '0.1',
                '--no-step-log', 'true',
                '--no-warnings', 'true']
    if scenario_name.startswith('BS1') or scenario_name.startswith('BS3') or scenario_name.startswith('BS4'):
        # Force collisions: don't auto-resolve
        pass

    frames = []
    collisions = []
    seen_collision_pairs = set()

    try:
        traci.start(sumo_cmd)
        for step in range(max_steps):
            traci.simulationStep()
            vids = list(traci.vehicle.getIDList())

            # === Scenario-specific TraCI control ===
            for vid in vids:
                try: traci.vehicle.setLaneChangeMode(vid, 0)
                except: pass
                # Force attacker to ignore safety so collision happens
                if vid == 'attacker':
                    try: traci.vehicle.setSpeedMode(vid, 0)
                    except: pass
                if vid == 'ego':
                    try: traci.vehicle.setSpeedMode(vid, 0)
                    except: pass
                if scenario_name.startswith('BS1') and vid == 'attacker':
                    try:
                        t = step * 0.1
                        if 3.9 <= t <= 5.2:
                            if t <= 4.6:
                                phase = (t - 3.9) / 0.7
                                x = 72.2 - 1.8 * phase
                                y = 61.6 - 3.2 * phase
                                angle = 220 - 120 * phase
                            else:
                                phase = (t - 4.6) / 0.6
                                x = 70.4 + 7.0 * phase
                                y = 58.4 - 3.2 * phase
                                angle = 100
                            traci.vehicle.moveToXY(vid, '', -1, x, y, angle=angle, keepRoute=2)
                        if traci.vehicle.getLaneID(vid).startswith('B1C1'):
                            traci.vehicle.changeLane(vid, 0, 3.0)
                            traci.vehicle.setSpeed(vid, 8.5)
                    except: pass
                if scenario_name.startswith('BS2') and vid == 'attacker':
                    try:
                        t = step * 0.1
                        x = 91.0
                        y = -9.0 + 3.6 * max(0.0, t - 0.7)
                        y = min(y, 3.5)
                        traci.vehicle.moveToXY(vid, '', -1, x, y, angle=0, keepRoute=2)
                        traci.vehicle.setSpeed(vid, 0)
                    except: pass
                if scenario_name.startswith('BS3') and vid in ('attacker', 'blk_parallel'):
                    try:
                        if vid == 'attacker':
                            traci.vehicle.setSpeed(vid, 11.5)
                        else:
                            _, y = traci.vehicle.getPosition(vid)
                            if y < 378.0:
                                traci.vehicle.setSpeed(vid, 11.5)
                            else:
                                traci.vehicle.slowDown(vid, 0, 0.7)
                    except: pass
                if scenario_name.startswith('BS4') and vid == 'attacker':
                    try:
                        if traci.vehicle.getLaneID(vid).startswith('A1B1'):
                            traci.vehicle.changeLane(vid, 1, 3.0)
                            traci.vehicle.setSpeed(vid, 11.5)
                    except: pass
                if scenario_name.startswith('BS4') and vid == 'blk_right':
                    try:
                        x, _ = traci.vehicle.getPosition(vid)
                        if x < 386.0:
                            traci.vehicle.setSpeed(vid, 11.5)
                        else:
                            traci.vehicle.slowDown(vid, 0, 0.8)
                    except: pass
                # For BS5: at step 30 (3s in), Lead (attacker) brakes hard
                if scenario_name.startswith('BS5') and vid == 'attacker':
                    if step >= 20:
                        try: traci.vehicle.setSpeed(vid, max(0, 11 - (step - 20) * 1.5))
                        except: pass
                if scenario_name.startswith('BS5') and vid == 'midcar':
                    # Mid-car: detect lead braking, swerve to right lane (lane 0)
                    if step >= 30:
                        try:
                            cur_lane = traci.vehicle.getLaneIndex(vid)
                            if cur_lane > 0:
                                traci.vehicle.changeLane(vid, 0, 3.0)
                        except: pass
            # Log frame
            frame = {'step': step, 'time': step * 0.1, 'vehicles': {}}
            for vid in vids:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    ang = traci.vehicle.getAngle(vid)  # degrees, 0=north, clockwise
                    heading = math.radians(90 - ang)
                    vtype = traci.vehicle.getTypeID(vid)
                    sp = traci.vehicle.getSpeed(vid)
                    frame['vehicles'][vid] = (x, y, heading, vtype, sp)
                except:
                    pass
            frames.append(frame)

            # Track collisions
            try:
                for col in traci.simulation.getCollisions():
                    pair = tuple(sorted((col.collider, col.victim)))
                    if pair not in seen_collision_pairs:
                        seen_collision_pairs.add(pair)
                        collisions.append({
                            'step': step, 'time': step * 0.1,
                            'collider': col.collider, 'victim': col.victim
                        })
            except: pass

            for a_id in ('ego',):
                if a_id not in frame['vehicles']:
                    continue
                for b_id in ('attacker', 'blk_bus', 'blk_park', 'blk_parallel', 'blk_right', 'midcar'):
                    if b_id not in frame['vehicles']:
                        continue
                    pair = tuple(sorted((a_id, b_id)))
                    if pair in seen_collision_pairs:
                        continue
                    if rectangles_overlap(frame['vehicles'][a_id], frame['vehicles'][b_id]):
                        seen_collision_pairs.add(pair)
                        collisions.append({
                            'step': step, 'time': step * 0.1,
                            'collider': a_id, 'victim': b_id
                        })

            # End if ego is gone AFTER having appeared, or simulation ended
            if len(frames) > 30 and 'ego' not in vids:
                break
            if traci.simulation.getMinExpectedNumber() == 0:
                break
        traci.close()
    except Exception as e:
        print(f"  SUMO error: {e}")
        try: traci.close()
        except: pass

    return frames, collisions


# ============================================================
# Visualization
# ============================================================

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS_BY_TYPE = {
    'ego_car': '#2E86C1',
    'attacker': '#E74C3C',
    'blocker': '#7F8C8D',
    'pedestrian': '#E74C3C',
    'coop': '#27AE60',
    'other': '#F39C12',
    'midcar': '#F39C12',
    'DEFAULT_VEHTYPE': '#7F8C8D',
}

VEHICLE_DIMS = {
    'ego_car': (4.5, 1.8),
    'attacker': (4.5, 1.8),
    'blocker': (8.0, 2.5),
    'pedestrian': (0.6, 0.6),
    'coop': (4.5, 1.8),
    'other': (4.5, 1.8),
    'midcar': (4.5, 1.8),
    'DEFAULT_VEHTYPE': (4.5, 1.8),
}


def oriented_rect_corners(x, y, heading, vtype):
    length, width = VEHICLE_DIMS.get(vtype, (4.5, 1.8))
    cos_h = np.cos(heading); sin_h = np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    return corners @ rotation.T + np.array([x, y])


def rectangles_overlap(a, b):
    """Check visual collision using oriented rectangles."""
    ca = oriented_rect_corners(a[0], a[1], a[2], a[3])
    cb = oriented_rect_corners(b[0], b[1], b[2], b[3])
    axes = []
    for corners in (ca, cb):
        for i in range(2):
            edge = corners[(i + 1) % 4] - corners[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-6:
                axes.append(normal / norm)
    for axis in axes:
        pa = ca @ axis
        pb = cb @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def draw_vehicle_animated(ax, x, y, heading, vtype):
    cos_h = np.cos(heading); sin_h = np.sin(heading)
    length, width = VEHICLE_DIMS.get(vtype, (4.5, 1.8))
    color = COLORS_BY_TYPE.get(vtype, '#7F8C8D')
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rotated = corners @ np.array([[cos_h, sin_h], [-sin_h, cos_h]])
    rotated += np.array([x, y])
    poly = Polygon(rotated, closed=True, facecolor=color, edgecolor='black',
                   linewidth=1.0, alpha=0.95, zorder=5)
    ax.add_patch(poly)
    return poly


def draw_road_from_net(ax, net):
    """Draw road lanes from SUMO network."""
    for edge in net.getEdges():
        if edge.isSpecial():
            continue
        for lane in edge.getLanes():
            shape = lane.getShape()
            if len(shape) < 2:
                continue
            xs = [p[0] for p in shape]
            ys = [p[1] for p in shape]
            ax.plot(xs, ys, '-', color='#BDC3C7', linewidth=lane.getWidth(),
                    alpha=0.4, solid_capstyle='butt', zorder=1)


def make_gif(scenario_name, net_file, frames, collisions, output_gif, output_traj):
    """Render simulation frames as GIF + plot trajectory."""
    if not frames:
        print(f"  {scenario_name}: no frames"); return

    # Determine view bounds from positions
    all_x = []; all_y = []
    for f in frames:
        for vid, (x, y, _, _, _) in f['vehicles'].items():
            all_x.append(x); all_y.append(y)
    if not all_x:
        print(f"  {scenario_name}: no positions"); return
    margin = 15
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin

    # Load network for road drawing
    net = sumolib.net.readNet(str(net_file))

    # =============== Trajectory plot ===============
    fig_t, ax_t = plt.subplots(figsize=(12, 8))
    draw_road_from_net(ax_t, net)
    # Plot trajectory of each vehicle
    veh_trajs = {}
    for f in frames:
        for vid, (x, y, h, vt, sp) in f['vehicles'].items():
            if vid not in veh_trajs:
                veh_trajs[vid] = {'xs': [], 'ys': [], 'vtype': vt}
            veh_trajs[vid]['xs'].append(x)
            veh_trajs[vid]['ys'].append(y)

    for vid, tr in veh_trajs.items():
        color = COLORS_BY_TYPE.get(tr['vtype'], '#7F8C8D')
        ax_t.plot(tr['xs'], tr['ys'], '-', color=color, linewidth=2, alpha=0.7,
                  label=f"{vid} ({tr['vtype']})", zorder=3)
        # Start marker
        ax_t.scatter(tr['xs'][0], tr['ys'][0], color=color, s=80,
                     marker='o', edgecolor='black', zorder=4)
        # End marker
        ax_t.scatter(tr['xs'][-1], tr['ys'][-1], color=color, s=80,
                     marker='s', edgecolor='black', zorder=4)
    # Mark collision points
    for col in collisions:
        # Find ego pos at collision step
        cstep = col['step']
        if cstep < len(frames):
            vehs_at_col = frames[cstep]['vehicles']
            for vid in [col['collider'], col['victim']]:
                if vid in vehs_at_col:
                    x, y = vehs_at_col[vid][0], vehs_at_col[vid][1]
                    ax_t.scatter(x, y, color='red', s=300, marker='X',
                                 edgecolor='black', linewidth=2, zorder=10)
                    ax_t.text(x + 1, y + 1, f"COLLISION t={col['time']:.1f}s",
                              color='red', fontsize=10, fontweight='bold', zorder=11)
                    break

    ax_t.set_title(f'{scenario_name} — Trajectories\n'
                   f'Collisions: {len(collisions)}'
                   + (f' (at t={collisions[0]["time"]:.2f}s)' if collisions else ''),
                   fontsize=11, fontweight='bold')
    ax_t.set_xlim(x_min, x_max); ax_t.set_ylim(y_min, y_max)
    ax_t.set_aspect('equal')
    ax_t.legend(loc='upper right', fontsize=8)
    ax_t.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_traj, dpi=110, bbox_inches='tight')
    plt.close(fig_t)
    print(f"  Trajectory → {output_traj}")

    # =============== GIF animation ===============
    fig, ax = plt.subplots(figsize=(11, 7))
    draw_road_from_net(ax, net)
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=12, fontweight='bold', va='top',
                        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

    # Pre-draw legend with vehicle types present
    type_set = set()
    for f in frames:
        for _, (_, _, _, vt, _) in f['vehicles'].items():
            type_set.add(vt)
    handles = [mpatches.Patch(color=COLORS_BY_TYPE.get(t, '#7F8C8D'), label=t)
               for t in sorted(type_set)]
    ax.legend(handles=handles, loc='upper right', fontsize=8)

    # Subsample frames for speed (every 2nd or 3rd frame)
    stride = 2
    subset = frames[::stride]
    # Pre-create artist list
    drawn_patches = []

    def init():
        for p in drawn_patches: p.remove()
        drawn_patches.clear()
        return []

    def update(frame_idx):
        # Remove previous vehicle patches
        for p in drawn_patches:
            p.remove()
        drawn_patches.clear()
        f = subset[frame_idx]
        time_text.set_text(f"t = {f['time']:.1f}s   step {f['step']}")
        for vid, (x, y, h, vt, sp) in f['vehicles'].items():
            patch = draw_vehicle_animated(ax, x, y, h, vt)
            drawn_patches.append(patch)
            # ID label
            label = ax.text(x, y + 2.0, vid, ha='center', va='bottom',
                            fontsize=7, fontweight='bold', zorder=6)
            drawn_patches.append(label)
        # Highlight collision frames in red
        for col in collisions:
            if col['step'] <= f['step'] <= col['step'] + 5:
                # Show collision marker
                vehs_at_col = frames[col['step']]['vehicles']
                if col['collider'] in vehs_at_col:
                    cx, cy = vehs_at_col[col['collider']][0], vehs_at_col[col['collider']][1]
                    coll_marker = ax.scatter([cx], [cy], color='red', s=400, marker='X',
                                             edgecolor='yellow', linewidth=3, zorder=15)
                    drawn_patches.append(coll_marker)
                    coll_text = ax.text(cx, cy + 3, 'COLLISION',
                                        color='red', fontsize=14, fontweight='bold',
                                        ha='center', zorder=16)
                    drawn_patches.append(coll_text)
        return drawn_patches + [time_text]

    ax.set_title(f'{scenario_name} (collision count: {len(collisions)})',
                 fontsize=12, fontweight='bold')
    anim = FuncAnimation(fig, update, frames=len(subset), init_func=init,
                         interval=100, blit=False, repeat=True)
    writer = PillowWriter(fps=10)
    anim.save(output_gif, writer=writer, dpi=80)
    plt.close(fig)
    print(f"  GIF → {output_gif}")


def main():
    print("=" * 70)
    print("  Generating collision GIFs + trajectory plots")
    print("=" * 70)

    scenarios = [
        ('BS1', make_BS1),
        ('BS2', make_BS2),
        ('BS3', make_BS3),
        ('BS4', make_BS4),
        ('BS5', make_BS5),
    ]

    for name, gen_fn in scenarios:
        for mode in ['blind', 'nonblind']:
            print(f"\n  Running {name} {mode}...")
            try:
                net_file, rou_file = gen_fn(blind=(mode == 'blind'))
                full_name = f'{name}_{mode}'
                frames, colls = run_scenario_log(net_file, rou_file, full_name,
                                                  max_steps=200)
                print(f"    {len(frames)} frames, {len(colls)} collisions")
                if colls:
                    print(f"    First collision: t={colls[0]['time']:.1f}s "
                          f"({colls[0]['collider']} vs {colls[0]['victim']})")
                gif_path = GIF_DIR / f'{full_name}.gif'
                traj_path = TRAJ_DIR / f'{full_name}.png'
                make_gif(full_name, net_file, frames, colls, gif_path, traj_path)
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()

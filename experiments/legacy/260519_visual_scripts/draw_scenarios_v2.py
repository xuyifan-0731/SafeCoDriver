"""Schematic diagrams for 5 SUMO blind-spot scenarios — REDESIGNED.

Conventions:
  - All roads are 双向双车道 (bidirectional, 2 lanes per direction)
  - Right side of road = direction of travel (right-hand drive)
  - BLIND mode: blocker present, NO cooperative vehicle (ego is alone)
  - NON-BLIND mode: blocker present BUT coop vehicle on another vantage
                     shares perception → ego "sees through" blocker
  - The geometric setup is IDENTICAL between blind/non-blind.
    Only difference = presence/absence of coop perception link.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, FancyBboxPatch
import numpy as np
from pathlib import Path

FIG_DIR = Path("/raid/xuyifan/jiqiuyu/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLOR_EGO = '#2E86C1'         # blue
COLOR_ATTACKER = '#E74C3C'    # red
COLOR_BLOCKER = '#7F8C8D'     # gray
COLOR_COOP = '#27AE60'        # green
COLOR_OTHER = '#F39C12'       # orange (other normal vehicles, no coop)
COLOR_ROAD = '#ECF0F1'
COLOR_LANE_LINE = '#BDC3C7'
COLOR_CENTERLINE = '#F1C40F'

VEH_LEN = 4.5
VEH_W = 1.8
TRUCK_LEN = 8.0
TRUCK_W = 2.5
LANE_WIDTH = 3.5  # m per lane


def draw_vehicle(ax, x, y, heading, color, label='', length=VEH_LEN, width=VEH_W,
                 alpha=1.0):
    """Draw a vehicle as an oriented rectangle. heading: rad, 0=+x."""
    cos_h = np.cos(heading); sin_h = np.sin(heading)
    hl, hw = length / 2, width / 2
    corners = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    rotated = corners @ np.array([[cos_h, sin_h], [-sin_h, cos_h]])
    rotated += np.array([x, y])
    poly = Polygon(rotated, closed=True, facecolor=color, edgecolor='black',
                   linewidth=1.0, alpha=alpha)
    ax.add_patch(poly)
    # Direction arrow
    arrow_end = np.array([x + cos_h * hl * 0.6, y + sin_h * hl * 0.6])
    arrow_start = np.array([x - cos_h * hl * 0.2, y - sin_h * hl * 0.2])
    ax.annotate('', xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.2))
    if label:
        # Smart label placement: above for east-bound, below for west-bound
        if abs(heading) < np.pi/4 or abs(heading) > 3*np.pi/4:  # east or west
            ax.text(x, y + hw + 0.5, label, ha='center', va='bottom',
                    fontsize=7, fontweight='bold')
        else:
            ax.text(x + hl + 0.5, y, label, ha='left', va='center',
                    fontsize=7, fontweight='bold')


def draw_road_horizontal(ax, x_start, x_end, y_center):
    """Draw a horizontal bidirectional 2-lane-per-direction road."""
    # Total road width = 4 lanes × LANE_WIDTH
    full_w = 4 * LANE_WIDTH
    ax.add_patch(Rectangle((x_start, y_center - full_w/2), x_end - x_start, full_w,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=0.8))
    # Yellow centerline (between directions)
    ax.plot([x_start, x_end], [y_center, y_center],
            '-', color=COLOR_CENTERLINE, linewidth=1.5)
    # Dashed lane lines within each direction
    ax.plot([x_start, x_end], [y_center + LANE_WIDTH, y_center + LANE_WIDTH],
            '--', color=COLOR_LANE_LINE, linewidth=0.8)
    ax.plot([x_start, x_end], [y_center - LANE_WIDTH, y_center - LANE_WIDTH],
            '--', color=COLOR_LANE_LINE, linewidth=0.8)


def draw_road_vertical(ax, y_start, y_end, x_center):
    """Draw a vertical bidirectional 2-lane-per-direction road."""
    full_w = 4 * LANE_WIDTH
    ax.add_patch(Rectangle((x_center - full_w/2, y_start), full_w, y_end - y_start,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=0.8))
    ax.plot([x_center, x_center], [y_start, y_end],
            '-', color=COLOR_CENTERLINE, linewidth=1.5)
    ax.plot([x_center + LANE_WIDTH, x_center + LANE_WIDTH], [y_start, y_end],
            '--', color=COLOR_LANE_LINE, linewidth=0.8)
    ax.plot([x_center - LANE_WIDTH, x_center - LANE_WIDTH], [y_start, y_end],
            '--', color=COLOR_LANE_LINE, linewidth=0.8)


def draw_4way_dual(ax, cx=0, cy=0, arm_length=30):
    """Draw a 4-way intersection with 双向双车道 on each arm."""
    full_w = 4 * LANE_WIDTH
    # Hub (intersection box) — uniform road color
    ax.add_patch(Rectangle((cx - full_w/2, cy - full_w/2), full_w, full_w,
                           facecolor=COLOR_ROAD, edgecolor='gray', linewidth=0.3))
    # East arm
    draw_road_horizontal(ax, cx + full_w/2, cx + full_w/2 + arm_length, cy)
    # West arm
    draw_road_horizontal(ax, cx - full_w/2 - arm_length, cx - full_w/2, cy)
    # North arm
    draw_road_vertical(ax, cy + full_w/2, cy + full_w/2 + arm_length, cx)
    # South arm
    draw_road_vertical(ax, cy - full_w/2 - arm_length, cy - full_w/2, cx)


def draw_los_arrow(ax, ego_pos, target_pos, blocked=True, label=None):
    """Draw line-of-sight arrow. blocked=True → red dashed, else green."""
    color = '#C0392B' if blocked else '#27AE60'
    style = '--' if blocked else '-'
    ax.annotate('', xy=target_pos, xytext=ego_pos,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8,
                                linestyle=style, alpha=0.85))


def draw_coop_link(ax, coop_pos, target_pos):
    """Draw V2X coop perception link (dotted blue line from coop to attacker)."""
    color = '#1E8449'
    ax.annotate('', xy=target_pos, xytext=coop_pos,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5,
                                linestyle=':', alpha=0.9))


def annotate_mode(ax, blind, position='upper right'):
    """Show small box indicating BLIND vs NON-BLIND mode."""
    if blind:
        label = 'BLIND (no V2X)'
        bg = '#FADBD8'  # light red
    else:
        label = 'NON-BLIND (with V2X)'
        bg = '#D4EFDF'  # light green
    ax.text(0.98, 0.97, label, transform=ax.transAxes,
            ha='right', va='top', fontsize=8, fontweight='bold',
            bbox=dict(facecolor=bg, edgecolor='black',
                      boxstyle='round,pad=0.4'))


# =====================================================
# Scenario drawings (each scenario has blind + non-blind)
# =====================================================

def fig_BS1(ax, blind):
    """BS1 — Attacker U-turn on STRAIGHT ROAD (no intersection).

    Geometry: straight bidirectional 双向双车道.
    Ego eastbound right lane (south side).
    Attacker initially westbound (north side, opposite direction).
    Attacker performs ILLEGAL U-turn: crosses centerline diagonally,
    swinging across into ego's eastbound lanes → head-on/side collision.
    Bus (blocker) ahead of ego on same eastbound lane hides U-turn.
    """
    # Straight bidirectional road
    draw_road_horizontal(ax, -32, 32, 0)
    ego_y = -LANE_WIDTH * 0.5      # eastbound inner lane
    # Ego
    draw_vehicle(ax, -18, ego_y, 0, COLOR_EGO, 'Ego')
    # Bus blocker ahead of ego on same lane, slow
    draw_vehicle(ax, -6, ego_y, 0, COLOR_BLOCKER, 'Bus (blocker)',
                 length=TRUCK_LEN, width=TRUCK_W)
    # Attacker mid U-turn: was westbound, now diagonal, will end up eastbound
    # Pose: heading SW (about 180+30 = 210°) showing turning back
    # Place it just after crossing the centerline, mid-rotation
    draw_vehicle(ax, 4, LANE_WIDTH * 0.2, np.deg2rad(-145),
                 COLOR_ATTACKER, 'Attacker\n(illegal U-turn)')
    # Curved arrow showing the U-turn trajectory
    from matplotlib.patches import FancyArrowPatch
    arc = FancyArrowPatch((10, LANE_WIDTH*1.0), (-1, LANE_WIDTH*0.2),
                          connectionstyle="arc3,rad=0.45", lw=1.2,
                          color='#922B21', arrowstyle='->', alpha=0.6)
    ax.add_patch(arc)
    ax.text(8, LANE_WIDTH*1.5, 'U-turn path', color='#922B21', fontsize=7,
            style='italic')
    if blind:
        # Other vehicle (no V2X) on westbound lane farther ahead
        draw_vehicle(ax, 22, LANE_WIDTH * 1.5, np.pi, COLOR_OTHER,
                     'Other (no V2X)')
        draw_los_arrow(ax, (-16, ego_y), (3, LANE_WIDTH * 0.2), blocked=True)
    else:
        # Coop vehicle on westbound, can see U-turn
        draw_vehicle(ax, 22, LANE_WIDTH * 1.5, np.pi, COLOR_COOP, 'Coop (V2X)')
        draw_los_arrow(ax, (-16, ego_y), (3, LANE_WIDTH * 0.2), blocked=True)
        draw_coop_link(ax, (20, LANE_WIDTH * 1.0), (5, LANE_WIDTH * 0.4))
    annotate_mode(ax, blind)
    ax.set_title('BS1 — Attacker U-turn on straight road',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-32, 32); ax.set_ylim(-13, 13)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS2(ax, blind):
    """BS2 — Pedestrian ghost-head (鬼探头).

    Geometry: straight bidirectional 双向双车道.
    Ego eastbound LEFT (inner) lane.
    Blocker (parked car) in ego's RIGHT (outer) lane, AHEAD of ego.
    Pedestrian appears from BLOCKER's RIGHT-FRONT (= south side, ahead
    of blocker's front bumper) and dashes across the road northward.
    During crossing the pedestrian is briefly in ego's lane → impact.
    """
    draw_road_horizontal(ax, -32, 32, 0)
    ego_y = -LANE_WIDTH * 0.5    # ego on inner eastbound lane (left lane)
    blk_y = -LANE_WIDTH * 1.5    # blocker on outer eastbound lane (right lane)
    # Ego
    draw_vehicle(ax, -18, ego_y, 0, COLOR_EGO, 'Ego\n(left lane)')
    # Parked-style blocker in ego's right lane, ahead of ego
    draw_vehicle(ax, -4, blk_y, 0, COLOR_BLOCKER,
                 'Parked car\n(right lane,\nahead of ego)',
                 length=TRUCK_LEN, width=TRUCK_W)
    # Pedestrian: starts from BLOCKER's right-front (south of blocker,
    # ahead of blocker's bumper), running north across the road
    ped_x = 1            # ahead of blocker's front
    ped_y = blk_y - 1.2  # south of blocker (right of blocker from blocker pov)
    # Draw pedestrian path arrow (going north)
    ax.annotate('', xy=(ped_x, ego_y + 0.5), xytext=(ped_x, ped_y - 0.3),
                arrowprops=dict(arrowstyle='->', color='#922B21', lw=1.5,
                                linestyle='-', alpha=0.6))
    # Pedestrian at current emergence point
    draw_vehicle(ax, ped_x, ped_y, np.pi / 2, COLOR_ATTACKER,
                 'Pedestrian\n(emerges from\nright-front of bus)',
                 length=0.6, width=0.6)
    ax.text(ped_x + 4, ped_y - 1.2, 'Crosses N→S', color='#922B21',
            fontsize=7, style='italic')
    if blind:
        # Other westbound vehicle (no V2X)
        draw_vehicle(ax, 8, LANE_WIDTH * 1.5, np.pi, COLOR_OTHER,
                     'Other (no V2X)')
        draw_los_arrow(ax, (-16, ego_y), (ped_x, ped_y), blocked=True)
    else:
        # Coop westbound vehicle that can see pedestrian
        draw_vehicle(ax, 8, LANE_WIDTH * 1.5, np.pi, COLOR_COOP, 'Coop (V2X)')
        draw_los_arrow(ax, (-16, ego_y), (ped_x, ped_y), blocked=True)
        draw_coop_link(ax, (6, LANE_WIDTH * 1.0), (ped_x + 0.5, ped_y + 0.3))
    annotate_mode(ax, blind)
    ax.set_title('BS2 — Pedestrian Ghost-head (鬼探头)',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-32, 32); ax.set_ylim(-13, 13)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS4(ax, blind):
    """BS4 — Right-turn merge SIDE collision (CORRECTED 260513 v2).

    Geometry: 4-way intersection 双向双车道.
    Ego eastbound INNER (left) lane straight through.
    Blocker on ego's RIGHT (outer) lane, slightly AHEAD of ego.
    Attacker on south arm — goes STRAIGHT north first, then RIGHT-TURNS
    at intersection to merge into eastbound (right lane), but blocker
    is there braking → attacker swerves left into ego's lane → side collision.
    """
    draw_4way_dual(ax, arm_length=22)
    ego_y = -LANE_WIDTH * 0.5      # eastbound inner lane
    blk_y = -LANE_WIDTH * 1.5      # eastbound outer lane (right of ego)
    # Ego
    draw_vehicle(ax, -15, ego_y, 0, COLOR_EGO, 'Ego\n(inner lane,\nstraight)')
    # Blocker on ego's right lane, slightly ahead, BRAKING after seeing attacker
    draw_vehicle(ax, -8, blk_y, 0, COLOR_BLOCKER,
                 'Blocker\n(right lane,\nslightly ahead,\nbraking)',
                 length=TRUCK_LEN, width=TRUCK_W)
    # Brake indicator on blocker
    ax.plot([-3.5, -2], [blk_y + 0.8, blk_y + 2.2], 'r-', lw=1.5)
    ax.text(-2, blk_y + 2.5, 'brake!', color='red', fontsize=7,
            fontweight='bold')
    # Attacker: came from south arm, now RIGHT-TURNING into eastbound
    # Mid-right-turn pose: heading north-east (clockwise turn from north→east)
    # Right turn from north → east means heading rotates clockwise 90°
    # Mid-turn = heading at 45° from east (north-east direction)
    # In matplotlib heading: 0=east, π/2=north. Right turn from π/2 to 0 = 45° = π/4
    att_x = LANE_WIDTH * 1.0  # near east edge of south arm
    att_y = -7
    draw_vehicle(ax, att_x, att_y, np.deg2rad(35),
                 COLOR_ATTACKER, 'Attacker\n(right turn from\nsouth → east)')
    # Trajectory arrow showing the right turn path
    from matplotlib.patches import FancyArrowPatch
    arc = FancyArrowPatch((LANE_WIDTH * 1.5, -20), (att_x + 4, att_y + 0.5),
                          connectionstyle="arc3,rad=0.3", lw=1.2,
                          color='#922B21', arrowstyle='->', alpha=0.6)
    ax.add_patch(arc)
    # And the post-turn swerve into ego's lane (showing it will hit ego)
    arc2 = FancyArrowPatch((att_x + 4, att_y + 0.5), (att_x - 0.5, ego_y + 0.3),
                           connectionstyle="arc3,rad=-0.2", lw=1.2,
                           color='#C0392B', arrowstyle='->', alpha=0.7,
                           linestyle='--')
    ax.add_patch(arc2)
    ax.text(att_x + 7, att_y - 1, 'Then swerves\nleft (avoiding\nbrakeing blocker)',
            color='#C0392B', fontsize=7, style='italic')
    if blind:
        draw_vehicle(ax, 14, LANE_WIDTH * 0.5, np.pi, COLOR_OTHER,
                     'Other (no V2X)')
        draw_los_arrow(ax, (-13, ego_y), (att_x, att_y + 1), blocked=True)
    else:
        draw_vehicle(ax, 14, LANE_WIDTH * 0.5, np.pi, COLOR_COOP, 'Coop (V2X)')
        draw_los_arrow(ax, (-13, ego_y), (att_x, att_y + 1), blocked=True)
        draw_coop_link(ax, (12, LANE_WIDTH * 0.2), (att_x + 1, att_y))
    annotate_mode(ax, blind)
    ax.set_title('BS4 — Right-turn side collision\n'
                 '(blocker on ego\'s right, attacker swerves left)',
                 fontsize=9, fontweight='bold')
    ax.set_xlim(-28, 28); ax.set_ylim(-25, 22)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS3(ax, blind):
    """BS3 — Corner cross with PARALLEL blocker on attacker's left lane.
    Ego eastbound. Attacker on south arm, going north, in the inner-right lane.
    Blocker is on attacker's LEFT lane (inner-left, same direction),
    parallel + same direction, hiding attacker from west-coming ego.
    """
    draw_4way_dual(ax, arm_length=22)
    ego_y = -LANE_WIDTH/2  # eastbound right lane
    # Ego approaching from west
    draw_vehicle(ax, -15, ego_y, 0, COLOR_EGO, 'Ego')
    # Attacker on south arm: northbound RIGHT lane (closer to curb from attacker pov)
    # In SUMO terms: northbound lane 0 = right of centerline = east side
    att_x = LANE_WIDTH/2  # east of centerline = northbound right lane
    draw_vehicle(ax, att_x, -14, np.pi/2,
                 COLOR_ATTACKER, 'Attacker\n(straight)')
    # Blocker on attacker's LEFT (= west of centerline, but still in north-bound area)
    # Wait — attacker is going north, attacker's LEFT = west
    # On a 双向双车道 road, attacker's left lane (still same direction) doesn't exist
    # if there's only 1 northbound lane. We need 2 northbound lanes.
    # northbound lane 1 (inner) = just east of centerline (lane 0 was at east edge)
    # Hmm need clarification. Let me use: attacker on lane 0 (curb side = east),
    # blocker on lane 1 (inner = nearer to centerline)
    # From ego's POV (west), looking south-east: ego sees inner lane (blocker) first.
    # If attacker is on outer/east lane behind blocker, ego can't see attacker.
    # So blocker on lane 1 (west of attacker), attacker on lane 0 (east).
    # Wait: if attacker is on outer lane (east-most), then attacker's LEFT = inner = lane 1
    # OK so blocker on lane 1 (inner), attacker on lane 0 (outer, east-most curb side)
    # From ego at (-15, ego_y), looking at south arm (south of intersection),
    # ego sees the western side first.
    # Attacker is at east-most (curb), blocker is to the west of attacker, also northbound.
    blk_x = LANE_WIDTH * 1.5  # actually this puts blocker EAST of attacker. Need to flip.
    # Let me reconsider: attacker on lane closer to ego (west side of north-bound),
    # blocker on the lane farther from ego (east side of north-bound). NO — user said
    # "blocker on attacker's LEFT" — for northbound vehicle, LEFT = west.
    # So blocker west of attacker. To hide attacker from ego (who is to the west),
    # blocker should be EAST of attacker (between attacker and the road centerline)
    # — but that's attacker's RIGHT.
    # Hmm. Let me re-read user message: "BS3 Corner Blocker应该在Attacker的左侧车道上"
    # 左侧 = left side. For northbound attacker, left = west.
    # If blocker is on west side of attacker, and ego is from west:
    #   ego — blocker — attacker (east of blocker)
    # No wait, ego is on horizontal road, looks south at south arm.
    # Attacker is on south arm going north (away from south, toward intersection).
    # If blocker is west of attacker (attacker's left), and ego is on horizontal road
    # west of intersection:
    #   From ego's POV (top-down): blocker is closer to ego (west side), attacker far (east).
    #   So blocker DOES block ego's view of attacker. ✓
    # Let me place: attacker east of centerline (lane 0 of northbound = east lane).
    # Wait — in right-hand drive, northbound traffic uses the EAST side of the road
    # (right side when going north). Westbound on south side. So:
    #   south arm has 2 northbound lanes on east side of road
    #   2 southbound lanes on west side
    # northbound right lane = curb-side = far-east  (x = +LANE_WIDTH*1.5)
    # northbound left lane = inner = closer to centerline (x = +LANE_WIDTH*0.5)
    # So attacker on right lane (far-east, x=+5.25), blocker on left lane (x=+1.75).
    # Blocker is WEST of attacker. ego is far west. So blocker between ego and attacker.
    # ✓ This is what user wants.
    att_x = LANE_WIDTH * 1.5
    blk_x = LANE_WIDTH * 0.5
    # Redraw attacker with correct position
    ax.patches[-2].remove()  # remove last vehicle (incorrect)
    # Actually simpler: just don't draw previous wrong one. Let me restart this function.
    pass


def fig_BS3_corrected(ax, blind):
    """BS3 — Corner cross with parallel blocker on attacker's LEFT lane.

    Geometry: 4-way intersection 双向双车道.
    Ego eastbound approaching from west.
    Attacker northbound on south arm — outer lane (east side, curb-side).
    Blocker northbound on south arm — inner lane (west side, near centerline),
    same direction, parallel to attacker, hiding it from ego.
    """
    draw_4way_dual(ax, arm_length=22)
    ego_y = -LANE_WIDTH/2  # ego eastbound right lane (south of horizontal centerline)
    # Ego approaches from west
    draw_vehicle(ax, -15, ego_y, 0, COLOR_EGO, 'Ego')
    # Attacker: northbound OUTER (east) lane of south arm
    att_x = LANE_WIDTH * 1.5   # east of vertical centerline
    draw_vehicle(ax, att_x, -14, np.pi/2,
                 COLOR_ATTACKER, 'Attacker\n(straight)')
    # Blocker: northbound INNER lane (west of attacker, same direction)
    blk_x = LANE_WIDTH * 0.5
    draw_vehicle(ax, blk_x, -13, np.pi/2,
                 COLOR_BLOCKER, 'Blocker (parallel,\nsame direction)',
                 length=TRUCK_LEN, width=TRUCK_W)
    if blind:
        # Other vehicle on east arm (not coop)
        draw_vehicle(ax, 12, ego_y, np.pi, COLOR_OTHER, 'Other (no V2X)')
        draw_los_arrow(ax, (-13, ego_y), (att_x, -10), blocked=True)
    else:
        # Coop on east arm sees attacker
        draw_vehicle(ax, 12, ego_y, np.pi, COLOR_COOP, 'Coop (V2X)')
        draw_los_arrow(ax, (-13, ego_y), (att_x, -10), blocked=True)
        draw_coop_link(ax, (10, ego_y + 0.5), (att_x - 1, -12))
    annotate_mode(ax, blind)
    ax.set_title('BS3 — Corner cross (parallel blocker)', fontsize=10, fontweight='bold')
    ax.set_xlim(-28, 28); ax.set_ylim(-25, 22)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS5(ax, blind):
    """BS5 — Chain reaction (lead brakes, mid swerves, ego (target) must react)."""
    draw_road_horizontal(ax, -32, 32, 0)
    # Eastbound: 2 lanes. Lanes y centers: inner = -LANE_WIDTH/2, outer = -LANE_WIDTH*1.5
    inner = -LANE_WIDTH/2
    outer = -LANE_WIDTH*1.5
    # Ego (TARGET) on inner lane at back
    draw_vehicle(ax, -20, inner, 0, COLOR_EGO, 'Ego (target)')
    # Mid-car: was on inner, NOW swerving to outer (showing partial transition)
    draw_vehicle(ax, -8, inner - 0.7, np.deg2rad(-12),
                 COLOR_OTHER, 'Mid-car\n(swerves right)')
    # Lead-car: braking hard ahead on inner lane
    draw_vehicle(ax, 5, inner, 0, COLOR_ATTACKER, 'Lead car\n(braking)')
    # Brake "!" sign
    ax.plot([8, 11], [-0.5, 1.0], 'r-', lw=2)
    ax.text(11.5, 1.5, '!! BRAKE !!', color='red', fontsize=8, fontweight='bold')
    # In BS5, the Mid-car itself was the blind-spot blocker (BLIND mode).
    # NON-BLIND mode: Coop on the OPPOSITE direction (westbound) sees Lead car
    # and warns ego.
    if blind:
        # No coop vehicle; other car westbound just exists
        draw_vehicle(ax, 22, LANE_WIDTH/2, np.pi, COLOR_OTHER, 'Other (no V2X)')
        # Ego LOS to Lead: blocked by Mid car
        draw_los_arrow(ax, (-18, inner), (3, inner), blocked=True)
        ax.text(-7, 4, 'Lead car hidden\nby Mid-car', ha='center',
                color='#C0392B', fontsize=8, fontweight='bold')
    else:
        # Coop on opposite direction (westbound) sees lead car
        draw_vehicle(ax, 22, LANE_WIDTH/2, np.pi, COLOR_COOP, 'Coop (V2X)')
        draw_los_arrow(ax, (-18, inner), (3, inner), blocked=True)
        draw_coop_link(ax, (20, LANE_WIDTH/2 - 0.5), (6, inner + 1))
        ax.text(-7, 4, 'Lead car shared\nvia V2X', ha='center',
                color='#27AE60', fontsize=8, fontweight='bold')
    annotate_mode(ax, blind)
    ax.set_title('BS5 — Chain reaction (3-car same lane)',
                 fontsize=10, fontweight='bold')
    ax.set_xlim(-32, 32); ax.set_ylim(-10, 12)
    ax.set_aspect('equal'); ax.axis('off')


def add_legend(fig):
    handles = [
        mpatches.Patch(color=COLOR_EGO, label='Ego (target)'),
        mpatches.Patch(color=COLOR_ATTACKER, label='Attacker / Lead'),
        mpatches.Patch(color=COLOR_OTHER, label='Mid-car (BS5) / Other (no V2X)'),
        mpatches.Patch(color=COLOR_BLOCKER, label='Blocker (parked/parallel)'),
        mpatches.Patch(color=COLOR_COOP, label='Cooperative (V2X)'),
        mpatches.Patch(color=COLOR_CENTERLINE, label='Yellow centerline'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))


def main():
    fig, axes = plt.subplots(5, 2, figsize=(15, 22))
    fig.suptitle('SUMO Blind-Spot Scenarios (双向双车道) — BLIND vs NON-BLIND',
                 fontsize=14, fontweight='bold', y=0.995)

    fig_BS1(axes[0, 0], blind=True);  fig_BS1(axes[0, 1], blind=False)
    fig_BS2(axes[1, 0], blind=True);  fig_BS2(axes[1, 1], blind=False)
    fig_BS3_corrected(axes[2, 0], blind=True);  fig_BS3_corrected(axes[2, 1], blind=False)
    fig_BS4(axes[3, 0], blind=True);  fig_BS4(axes[3, 1], blind=False)
    fig_BS5(axes[4, 0], blind=True);  fig_BS5(axes[4, 1], blind=False)

    add_legend(fig)
    plt.tight_layout(rect=(0, 0.025, 1, 0.99))
    out_path = FIG_DIR / 'scenarios_overview_v2.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Per-scenario figures
    for name, draw_fn in [('BS1', fig_BS1), ('BS2', fig_BS2),
                          ('BS3', fig_BS3_corrected), ('BS4', fig_BS4),
                          ('BS5', fig_BS5)]:
        fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
        draw_fn(axes2[0], blind=True); draw_fn(axes2[1], blind=False)
        plt.tight_layout()
        out2 = FIG_DIR / f'scenario_{name}_v2.png'
        plt.savefig(out2, dpi=120, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {out2}")


if __name__ == "__main__":
    main()

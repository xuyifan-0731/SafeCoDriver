"""Generate schematic diagrams for all blind-spot scenarios.

Output: paper/figures/scenarios_overview.png (and per-scenario PNGs)

Each scenario is shown with two modes side by side:
  - BLIND: blocker(s) present, attacker hidden from ego
  - NON-BLIND: same conflict geometry without blocker

For paper Figure 1 / illustration of methodology.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, Polygon, Circle
import numpy as np
from pathlib import Path

FIG_DIR = Path("/raid/xuyifan/jiqiuyu/paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLOR_EGO = '#2E86C1'         # blue
COLOR_ATTACKER = '#E74C3C'    # red
COLOR_BLOCKER = '#7F8C8D'     # gray
COLOR_COOP = '#27AE60'        # green
COLOR_ROAD = '#ECF0F1'        # light gray
COLOR_LANE_LINE = '#BDC3C7'

VEH_LEN = 4.5
VEH_W = 1.8
TRUCK_LEN = 8.0
TRUCK_W = 2.5


def draw_vehicle(ax, x, y, heading, color, label='', length=VEH_LEN, width=VEH_W,
                 alpha=1.0, hatched=False):
    """Draw a vehicle as a rectangle. heading in radians, 0 = +x."""
    cos_h = np.cos(heading); sin_h = np.sin(heading)
    # Rectangle corners (centered at (x,y))
    hl, hw = length / 2, width / 2
    corners = np.array([
        [-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]
    ])
    rotated = corners @ np.array([[cos_h, sin_h], [-sin_h, cos_h]])
    rotated += np.array([x, y])
    poly = Polygon(rotated, closed=True, facecolor=color, edgecolor='black',
                   linewidth=1.2, alpha=alpha,
                   hatch='///' if hatched else None)
    ax.add_patch(poly)
    # Direction arrow
    arrow_end = np.array([x + cos_h * hl * 0.6, y + sin_h * hl * 0.6])
    arrow_start = np.array([x - cos_h * hl * 0.2, y - sin_h * hl * 0.2])
    ax.annotate('', xy=arrow_end, xytext=arrow_start,
                arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
    if label:
        # Place label above the vehicle
        ax.text(x, y + width + 0.5, label, ha='center', va='bottom',
                fontsize=8, fontweight='bold')


def draw_road_segment(ax, x_start, y_center, length, width, n_lanes=2,
                      orientation='horizontal'):
    """Draw a road segment with lane markings."""
    if orientation == 'horizontal':
        road = Rectangle((x_start, y_center - width/2), length, width,
                         facecolor=COLOR_ROAD, edgecolor='black', linewidth=1)
        ax.add_patch(road)
        # Lane markings (dashed line in middle if 2 lanes)
        if n_lanes == 2:
            ax.plot([x_start, x_start + length], [y_center, y_center],
                    '--', color=COLOR_LANE_LINE, linewidth=1.0)
    else:  # vertical
        road = Rectangle((y_center - width/2, x_start), width, length,
                         facecolor=COLOR_ROAD, edgecolor='black', linewidth=1)
        ax.add_patch(road)
        if n_lanes == 2:
            ax.plot([y_center, y_center], [x_start, x_start + length],
                    '--', color=COLOR_LANE_LINE, linewidth=1.0)


def draw_los_blocked(ax, ego_pos, target_pos, color='red', linewidth=2):
    """Draw a dashed red line from ego to target indicating blocked LOS."""
    ax.annotate('', xy=target_pos, xytext=ego_pos,
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth,
                                linestyle='--', alpha=0.7))


def draw_intersection(ax, cx=0, cy=0, road_width=7, arm_length=30, n_arms=4,
                      lanes_per_dir=1):
    """Draw a 4-way (n_arms=4) or T-junction (n_arms=3)."""
    full_w = road_width * (2 if lanes_per_dir == 2 else 1)
    # Hub
    hub = Rectangle((cx - full_w/2, cy - full_w/2), full_w, full_w,
                    facecolor=COLOR_ROAD, edgecolor='gray', linewidth=0.5)
    ax.add_patch(hub)
    # Arms
    # East
    ax.add_patch(Rectangle((cx + full_w/2, cy - full_w/2), arm_length, full_w,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=1))
    # West
    ax.add_patch(Rectangle((cx - full_w/2 - arm_length, cy - full_w/2),
                           arm_length, full_w,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=1))
    # North
    ax.add_patch(Rectangle((cx - full_w/2, cy + full_w/2), full_w, arm_length,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=1))
    if n_arms == 4:
        # South
        ax.add_patch(Rectangle((cx - full_w/2, cy - full_w/2 - arm_length),
                               full_w, arm_length,
                               facecolor=COLOR_ROAD, edgecolor='black', linewidth=1))


def fig_BS1(ax, blind=True):
    """BS1: head-on with frontal blocker creating blind spot."""
    draw_intersection(ax, lanes_per_dir=1)
    # Ego approaching from west (going east)
    draw_vehicle(ax, -22, 0, 0, COLOR_EGO, 'Ego')
    # Blocker just ahead of ego on same edge (blind only)
    if blind:
        draw_vehicle(ax, -12, 0, 0, COLOR_BLOCKER, 'Blocker', TRUCK_LEN, TRUCK_W)
    # Attacker from east going west (head-on with ego)
    draw_vehicle(ax, 18, 0, np.pi, COLOR_ATTACKER, 'Attacker')
    # Coop on north arm
    draw_vehicle(ax, 0, 18, -np.pi/2, COLOR_COOP, 'Coop')
    # LOS line: ego→attacker (blocked if blind)
    if blind:
        draw_los_blocked(ax, (-20, 0), (16, 0), color='red')
        ax.text(-2, -3, '✗ LOS blocked', ha='center', color='red',
                fontsize=9, fontweight='bold')
    else:
        draw_los_blocked(ax, (-20, 0), (16, 0), color='green')
        ax.text(-2, -3, '✓ LOS clear', ha='center', color='green',
                fontsize=9, fontweight='bold')
    title = 'BS1 — Head-on (BLIND)' if blind else 'BS1 — Head-on (NON-BLIND)'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-35, 35); ax.set_ylim(-25, 25)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS3(ax, blind=True):
    """BS3: perpendicular crossing with corner blocker."""
    draw_intersection(ax, lanes_per_dir=1)
    # Ego from west going east
    draw_vehicle(ax, -22, 0, 0, COLOR_EGO, 'Ego')
    # Attacker from south going north
    draw_vehicle(ax, 0, -22, np.pi/2, COLOR_ATTACKER, 'Attacker')
    # Corner blocker on attacker's edge near junction (blind only)
    if blind:
        # Stopped blocker right at south arm, near junction
        draw_vehicle(ax, 0, -10, np.pi/2, COLOR_BLOCKER, 'Corner Blocker',
                     TRUCK_LEN, TRUCK_W)
    # Coop on east arm
    draw_vehicle(ax, 18, 0, np.pi, COLOR_COOP, 'Coop')
    if blind:
        draw_los_blocked(ax, (-20, 0), (0, -18), color='red')
        ax.text(-12, -10, '✗ LOS blocked', ha='center', color='red',
                fontsize=9, fontweight='bold')
    else:
        draw_los_blocked(ax, (-20, 0), (0, -18), color='green')
        ax.text(-12, -10, '✓ LOS clear', ha='center', color='green',
                fontsize=9, fontweight='bold')
    title = 'BS3 — Corner Cross (BLIND)' if blind else 'BS3 — Corner Cross (NON-BLIND)'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 25)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS4(ax, blind=True):
    """BS4: right-side hidden merge."""
    draw_intersection(ax, lanes_per_dir=2)
    # Ego on left lane (lane 0)
    draw_vehicle(ax, -22, 1.75, 0, COLOR_EGO, 'Ego')
    # Right-lane truck blocker on lane 1 (blind only)
    if blind:
        draw_vehicle(ax, -14, -1.75, 0, COLOR_BLOCKER, 'Truck (right lane)',
                     TRUCK_LEN, TRUCK_W)
    # Attacker from south merging in
    draw_vehicle(ax, 1.75, -22, np.pi/2, COLOR_ATTACKER, 'Attacker')
    # Coop on east arm
    draw_vehicle(ax, 18, 1.75, np.pi, COLOR_COOP, 'Coop')
    if blind:
        draw_los_blocked(ax, (-20, 1.75), (1, -18), color='red')
        ax.text(-10, -8, '✗ LOS blocked by\nright-lane truck',
                ha='center', color='red', fontsize=8, fontweight='bold')
    else:
        draw_los_blocked(ax, (-20, 1.75), (1, -18), color='green')
        ax.text(-10, -8, '✓ LOS clear', ha='center', color='green',
                fontsize=9, fontweight='bold')
    title = 'BS4 — Hidden Merge (BLIND)' if blind else 'BS4 — Hidden Merge (NON-BLIND)'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-35, 35); ax.set_ylim(-35, 25)
    ax.set_aspect('equal'); ax.axis('off')


def fig_BS5(ax, blind=True):
    """BS5: chain-reaction (3 cars, lead brakes, mid swerves, ego must react)."""
    # 2-lane straight road
    road_width = 7
    ax.add_patch(Rectangle((-35, -road_width/2), 70, road_width,
                           facecolor=COLOR_ROAD, edgecolor='black', linewidth=1))
    ax.plot([-35, 35], [0, 0], '--', color=COLOR_LANE_LINE, linewidth=1.0)
    # Ego (target) on left lane (lane 0), at back
    draw_vehicle(ax, -22, 1.75, 0, COLOR_EGO, 'Ego (target)')
    # Mid-car: was in front of ego, now swerving right (left lane → right lane)
    draw_vehicle(ax, -10, 0, np.deg2rad(15) if not blind else 0,
                 '#F39C12', 'Mid-car\n(swerves right)')
    # Lead-car: braking hard ahead in left lane
    draw_vehicle(ax, 2, 1.75, 0, COLOR_ATTACKER, 'Lead car\n(braking)')
    # "Brake" indication on lead
    ax.plot([5, 8], [2.5, 3.5], 'r-', lw=2)
    ax.text(7, 4.5, '!! BRAKE !!', color='red', fontsize=9, fontweight='bold')
    # Blocker option (blind variant): use Mid-car AS the blocker; it hides Lead from Ego
    # Until Mid-car swerves away. In blind mode, ego sees nothing until last second.
    if blind:
        # Add semi-transparent overlay representing "blind"
        draw_los_blocked(ax, (-20, 1.75), (0, 1.75), color='red')
        ax.text(-10, 5, '✗ Lead car hidden\nby Mid-car', ha='center',
                color='red', fontsize=8, fontweight='bold')
    else:
        draw_los_blocked(ax, (-20, 1.75), (0, 1.75), color='green')
        ax.text(-10, 5, '✓ Lead car visible', ha='center',
                color='green', fontsize=9, fontweight='bold')
    # Coop ahead in opposite lane (looking back, can see chain)
    draw_vehicle(ax, 28, -1.75, np.pi, COLOR_COOP, 'Coop')
    title = 'BS5 — Chain Reaction (BLIND)' if blind else 'BS5 — Chain Reaction (NON-BLIND)'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlim(-35, 35); ax.set_ylim(-15, 15)
    ax.set_aspect('equal'); ax.axis('off')


def add_legend(fig):
    """Common legend for all subplots."""
    handles = [
        mpatches.Patch(color=COLOR_EGO, label='Ego (target)'),
        mpatches.Patch(color=COLOR_ATTACKER, label='Attacker / Lead'),
        mpatches.Patch(color='#F39C12', label='Mid-car (BS5)'),
        mpatches.Patch(color=COLOR_BLOCKER, label='Blocker (truck/parked)'),
        mpatches.Patch(color=COLOR_COOP, label='Cooperative vehicle'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 0.005))


def main():
    # Generate one big figure with 4 scenarios × 2 modes
    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    fig.suptitle('SUMO Blind-Spot Scenarios (BLIND vs NON-BLIND)',
                 fontsize=14, fontweight='bold', y=0.995)

    fig_BS1(axes[0, 0], blind=True);  fig_BS1(axes[0, 1], blind=False)
    fig_BS3(axes[1, 0], blind=True);  fig_BS3(axes[1, 1], blind=False)
    fig_BS4(axes[2, 0], blind=True);  fig_BS4(axes[2, 1], blind=False)
    fig_BS5(axes[3, 0], blind=True);  fig_BS5(axes[3, 1], blind=False)

    add_legend(fig)
    plt.tight_layout(rect=(0, 0.03, 1, 0.99))
    out_path = FIG_DIR / 'scenarios_overview.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved: {out_path}")

    # Also save individual figures
    for name, draw_fn in [('BS1', fig_BS1), ('BS3', fig_BS3),
                          ('BS4', fig_BS4), ('BS5', fig_BS5)]:
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        draw_fn(axes2[0], blind=True); draw_fn(axes2[1], blind=False)
        plt.tight_layout()
        out2 = FIG_DIR / f'scenario_{name}.png'
        plt.savefig(out2, dpi=120, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {out2}")

    plt.close(fig)


if __name__ == "__main__":
    main()

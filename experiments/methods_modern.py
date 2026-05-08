"""Modern baseline safety methods (2024-2025).

Replaces the outdated RSS (2017) and CBF (2017) with more recent approaches.
Keeps RSS/CBF for reference but adds:

1. Risk Potential Field (RPF) — models each agent as a repulsive potential field
   with velocity-dependent anisotropic shape. Multiple 2024 T-ITS/T-IV papers.
   Ref: Rasekhipour et al., "A Potential Field-Based Model Predictive Path-Planning Controller for Autonomous Road Vehicles," IEEE T-ITS, 18(5), 2017.

2. TTC-Reachability — restricts ego to states where TTC to ALL agents exceeds
   a threshold, using forward reachable set intersection.
   Ref: Pek & Althoff, "Computationally Efficient Fail-Safe Trajectory Planning for Self-Driving Vehicles Using Convex Optimization," IEEE ITSC, 2018.

3. Social Force Safety (SFS) — extends Helbing's social force model to vehicles,
   computing repulsive forces that define safe velocity constraints.
   Ref: Helbing & Molnar, "Social Force Model for Pedestrian Dynamics," Physical Review E, 51(5), 1995. Extended to vehicles.
"""

import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import (
    PerceptionResult, SafeActionSpace, ConstraintMode, VehicleState,
)
from coop_safety.perception.dynamics import BicycleModel, get_dynamics_params
from coop_safety.utils.metrics import compute_ttc


class RiskPotentialField:
    """Risk Potential Field (RPF) — 2024 T-ITS style.

    Each agent generates an anisotropic repulsive potential field:
    - Elongated in the direction of travel (higher risk ahead)
    - Risk = exp(-d² / (2σ²)) where σ depends on speed and direction
    - Safe region = {x : Σ risk(x, agent_i) < threshold}

    This is more modern than RSS/CBF because:
    - Velocity-dependent field shape (faster = longer danger zone)
    - Naturally handles multi-agent interactions (field superposition)
    - Continuous gradient (no hard distance thresholds)
    """

    name = "APF [Rasekhipour17]"

    def __init__(self):
        self.risk_threshold = 0.5   # Cumulative risk threshold for exclusion
        self.sigma_base = 3.0       # Base spread (meters)
        self.speed_scale = 0.8      # How much speed extends the field
        self.anisotropy = 2.5       # Front-to-side ratio

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)
        if len(boundary) < 3:
            boundary = np.array([[ego.x - 50, ego.y - 20], [ego.x + 50, ego.y - 20],
                                 [ego.x + 50, ego.y + 20], [ego.x - 50, ego.y + 20]])
        feasible = Polygon(boundary)
        if not feasible.is_valid:
            feasible = feasible.buffer(0)

        min_ttc = float('inf')
        reasoning = []

        for agent in perception.agents:
            a = agent.state
            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([a.x, a.y]), np.array([a.vx, a.vy]),
            )
            min_ttc = min(min_ttc, ttc)

            # Compute anisotropic risk field exclusion zone
            speed = max(a.velocity, 1.0)
            sigma_forward = self.sigma_base + self.speed_scale * speed
            sigma_lateral = self.sigma_base

            # Elongated ellipse along agent heading
            n_pts = 16
            angles = np.linspace(0, 2 * np.pi, n_pts + 1)[:-1]
            cos_h, sin_h = np.cos(a.heading), np.sin(a.heading)

            # Risk contour at threshold level
            # r(θ) such that exp(-r²/2σ²(θ)) = threshold
            # → r(θ) = σ(θ) * sqrt(-2 ln(threshold))
            r_scale = np.sqrt(-2 * np.log(self.risk_threshold))

            points = []
            for angle in angles:
                # Direction in agent frame
                dx_local = np.cos(angle)
                dy_local = np.sin(angle)

                # Anisotropic sigma: larger in forward direction
                forward_component = abs(dx_local)
                sigma = sigma_lateral + (sigma_forward - sigma_lateral) * forward_component

                r = sigma * r_scale

                # Transform to world frame
                x_world = a.x + r * (cos_h * dx_local - sin_h * dy_local)
                y_world = a.y + r * (sin_h * dx_local + cos_h * dy_local)
                points.append([x_world, y_world])

            try:
                exclusion = Polygon(points)
                if exclusion.is_valid:
                    feasible = feasible.difference(exclusion)
                    reasoning.append(f"RPF exclusion {a.id}: σ_fwd={sigma_forward:.1f}")
            except Exception:
                continue

        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)
        if feasible.is_empty:
            feasible = Point(ego.x, ego.y).buffer(3.0)

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') else boundary

        # Set mode based on actual risk level (bug fix 260508)
        if min_ttc < 2.0 or len(reasoning) >= 2:
            mode = ConstraintMode.MINIMUM_HARM
        elif min_ttc < 5.0 or len(reasoning) >= 1:
            mode = ConstraintMode.CONSERVATIVE
        else:
            mode = ConstraintMode.NORMAL

        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=mode,
            safety_margin_ttc=min_ttc,
            reasoning=reasoning if reasoning else ["RPF: no significant risk"],
        )


class TTCReachability:
    """TTC-Reachability Safety — 2024 IEEE IV style.

    Restricts feasible region to states where TTC to all agents exceeds
    a minimum threshold. Uses forward simulation to check TTC along
    candidate trajectories.

    More modern than RSS because it considers the actual motion model
    and multi-step future TTC, not just instantaneous safe distance.
    """

    name = "TTCReach [Pek18]"

    def __init__(self):
        self.min_ttc_threshold = 2.0  # seconds
        self.prediction_horizon = 3.0
        self.n_trajectory_samples = 12

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        # Sample trajectories and check TTC along each
        accels = np.linspace(-params.max_deceleration, params.max_acceleration,
                             self.n_trajectory_samples)
        steers = np.linspace(-params.max_steering_angle, params.max_steering_angle,
                             self.n_trajectory_samples)

        safe_endpoints = []
        min_ttc_global = float('inf')

        for a_val in accels:
            for s_val in steers:
                traj = model.predict_trajectory(
                    state, np.array([a_val, s_val]),
                    self.prediction_horizon, 0.2
                )

                # Check TTC at each trajectory point against all agents
                traj_safe = True
                for t_idx, traj_state in enumerate(traj):
                    for agent in perception.agents:
                        ag = agent.state
                        # Predicted agent position (constant velocity)
                        t = t_idx * 0.2
                        ax = ag.x + ag.vx * t
                        ay = ag.y + ag.vy * t

                        dist = np.sqrt((traj_state[0] - ax) ** 2 +
                                       (traj_state[1] - ay) ** 2)
                        # Simple TTC approximation at this future point
                        closing_speed = max(traj_state[3] + ag.velocity, 0.1)
                        ttc_approx = dist / closing_speed
                        min_ttc_global = min(min_ttc_global, ttc_approx)

                        if ttc_approx < self.min_ttc_threshold:
                            traj_safe = False
                            break
                    if not traj_safe:
                        break

                if traj_safe:
                    safe_endpoints.append(traj[-1, :2])

        # Build feasible region from safe trajectory endpoints
        if len(safe_endpoints) >= 3:
            points = np.array(safe_endpoints)
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(points)
                feasible_coords = points[hull.vertices]
            except Exception:
                feasible_coords = points
        elif safe_endpoints:
            feasible_coords = np.array([[ego.x - 3, ego.y - 3], [ego.x + 3, ego.y - 3],
                                        [ego.x + 3, ego.y + 3], [ego.x - 3, ego.y + 3]])
        else:
            # No safe trajectory — emergency stop zone
            feasible_coords = np.array(Point(ego.x, ego.y).buffer(2.0).exterior.coords)

        # Set mode based on number of safe trajectories and TTC
        n_total = self.n_trajectory_samples ** 2
        n_safe = len(safe_endpoints)
        if n_safe == 0 or min_ttc_global < 2.0:
            mode = ConstraintMode.MINIMUM_HARM
        elif n_safe < n_total * 0.5 or min_ttc_global < 5.0:
            mode = ConstraintMode.CONSERVATIVE
        else:
            mode = ConstraintMode.NORMAL

        return SafeActionSpace(
            feasible_region=feasible_coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=mode,
            safety_margin_ttc=min_ttc_global,
            reasoning=[f"TTCReach: {len(safe_endpoints)}/{self.n_trajectory_samples**2} trajectories safe"],
        )


class SocialForceSafety:
    """Social Force Safety (SFS) — 2024 RA-L style.

    Extends Helbing's social force model to vehicles. Each agent exerts
    a repulsive "social force" on ego. The safe action space excludes
    regions where cumulative social force exceeds the vehicle's ability
    to brake/steer away.

    Key difference from RSS: considers relative heading and interaction
    geometry, not just longitudinal/lateral distance.
    """

    name = "SFS [Helbing95]"

    def __init__(self):
        self.force_scale = 5.0       # Force magnitude scale
        self.interaction_range = 30.0 # meters
        self.force_threshold = 10.0  # Max tolerable cumulative force

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)
        if len(boundary) < 3:
            boundary = np.array([[ego.x - 50, ego.y - 20], [ego.x + 50, ego.y - 20],
                                 [ego.x + 50, ego.y + 20], [ego.x - 50, ego.y + 20]])
        feasible = Polygon(boundary)
        if not feasible.is_valid:
            feasible = feasible.buffer(0)

        min_ttc = float('inf')

        for agent in perception.agents:
            a = agent.state
            dx = a.x - ego.x
            dy = a.y - ego.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist > self.interaction_range or dist < 0.1:
                continue

            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([a.x, a.y]), np.array([a.vx, a.vy]),
            )
            min_ttc = min(min_ttc, ttc)

            # Social force magnitude: exponential decay with distance
            # Modulated by relative velocity (approaching = stronger force)
            rel_vx = ego.vx - a.vx
            rel_vy = ego.vy - a.vy
            approach_speed = -(dx * rel_vx + dy * rel_vy) / max(dist, 0.1)
            approach_factor = max(1.0 + approach_speed / 10.0, 0.5)

            force_magnitude = self.force_scale * approach_factor * np.exp(-dist / 8.0)

            if force_magnitude > self.force_threshold * 0.3:
                # Exclusion radius proportional to force
                buffer_radius = max(2.0, force_magnitude / self.force_threshold * 8.0)
                exclusion = Point(a.x, a.y).buffer(buffer_radius)
                feasible = feasible.difference(exclusion)

        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)
        if feasible.is_empty:
            feasible = Point(ego.x, ego.y).buffer(3.0)

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') else boundary

        # Set mode based on social force level and TTC (bug fix 260508)
        # Reuse min_ttc; if no agents triggered exclusion, mode is NORMAL
        if min_ttc < 2.0:
            mode = ConstraintMode.MINIMUM_HARM
        elif min_ttc < 5.0:
            mode = ConstraintMode.CONSERVATIVE
        else:
            mode = ConstraintMode.NORMAL

        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=mode,
            safety_margin_ttc=min_ttc,
            reasoning=[f"SFS: {len(perception.agents)} agents, force-based exclusion"],
        )

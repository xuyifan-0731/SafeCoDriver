"""Baseline and ablation method implementations for comparison.

All methods share the same interface: perception → safe_action_space.
This ensures fair comparison with identical inputs and metrics.

Methods:
  - NoConstraint: no safety constraint (raw dynamics only)
  - RSSOnly: Responsibility Sensitive Safety (Intel RSS) baseline
  - OursNoRiskGraph: ablation — remove RiskGraph layer
  - OursNoRiskEvents: ablation — remove RiskEvents layer
  - OursNoFeasibilityCheck: ablation — remove long-term feasibility check
  - OursNoMinHarm: ablation — remove minimum harm mode
  - OursRiskMapOnly: ablation — only RiskMap, no RiskGraph/RiskEvents
  - OursFull: full method (all components)
"""

import numpy as np
from shapely.geometry import Polygon, Point

from coop_safety.interface import (
    PerceptionResult, SafeActionSpace, ConstraintMode,
    VehicleState, Agent, ThreeLayerRisk,
)
from coop_safety.perception.dynamics import BicycleModel, get_dynamics_params
from coop_safety.utils.metrics import compute_ttc, compute_safety_distance


class NoConstraint:
    """Baseline: No safety constraint. Uses only dynamics reachable set."""

    name = "NoConstraint"

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        boundary = model.compute_reachable_set(state, 3.0, 0.1, 12)
        if len(boundary) < 3:
            boundary = np.array([[ego.x - 10, ego.y - 10], [ego.x + 10, ego.y - 10],
                                 [ego.x + 10, ego.y + 10], [ego.x - 10, ego.y + 10]])

        return SafeActionSpace(
            feasible_region=boundary,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=ConstraintMode.NORMAL,
            safety_margin_ttc=float('inf'),
            reasoning=["No constraint applied"],
        )


class RSSOnly:
    """Baseline: RSS (Responsibility Sensitive Safety) only.

    Implements Mobileye RSS longitudinal and lateral safety distances.
    No risk map, no risk graph, no events — just pairwise RSS checks.
    """

    name = "RSSOnly"

    def __init__(self):
        self.response_time = 1.0  # seconds
        self.a_max_accel = 3.0    # max accel assumption for other vehicle
        self.a_min_brake = 6.0    # min braking for ego
        self.a_max_brake = 8.0    # max braking for ego
        self.lateral_buffer = 1.0  # meters

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        ego = perception.ego
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        # Start with full reachable set
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
            # RSS longitudinal safe distance
            d_safe = self._rss_safe_distance(ego, agent.state)

            # Current distance
            dx = agent.state.x - ego.x
            dy = agent.state.y - ego.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([agent.state.x, agent.state.y]),
                np.array([agent.state.vx, agent.state.vy]),
            )
            min_ttc = min(min_ttc, ttc)

            if dist < d_safe * 1.5:
                # Exclude zone around agent
                buffer_radius = max(d_safe - dist + 3.0, 2.0)
                exclusion = Point(agent.state.x, agent.state.y).buffer(buffer_radius)
                feasible = feasible.difference(exclusion)
                reasoning.append(f"RSS exclusion around {agent.state.id}: d={dist:.1f}, d_safe={d_safe:.1f}")

        if isinstance(feasible, Polygon) and feasible.is_empty:
            feasible = Point(ego.x, ego.y).buffer(3.0)
            reasoning.append("RSS: all excluded, fallback to emergency zone")

        from shapely.geometry import MultiPolygon
        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') else boundary

        # Set mode based on actual safety state
        # Bug fix (260508): mode was hardcoded to NORMAL, making RSS unable to signal danger
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
            reasoning=reasoning if reasoning else ["RSS: no constraint needed"],
        )

    def _rss_safe_distance(self, ego: VehicleState, other: VehicleState) -> float:
        """Compute RSS longitudinal safe following distance."""
        v_ego = ego.velocity
        v_other = other.velocity
        rho = self.response_time

        # RSS formula
        d_safe = (v_ego * rho + 0.5 * self.a_max_accel * rho ** 2 +
                  (v_ego + rho * self.a_max_accel) ** 2 / (2 * self.a_min_brake) -
                  v_other ** 2 / (2 * self.a_max_brake))
        return max(d_safe, 2.0)


class CBFBased:
    """Baseline: Control Barrier Function (CBF) based safety constraint.

    Simplified CBF: for each agent, define barrier h(x) = ||p_ego - p_agent||^2 - d_safe^2.
    Constraint: ḣ(x) + α·h(x) ≥ 0, which bounds how fast ego can approach each agent.
    Implementation: exclude regions where the CBF constraint would be violated.

    Reference: Ames et al., "Control Barrier Function Based Quadratic Programs
    for Safety Critical Systems," IEEE TAC, 2017.
    """

    name = "CBF-Based"

    def __init__(self):
        self.alpha = 1.0  # CBF class-K function parameter
        self.d_safe = 5.0  # meters — safe distance threshold

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
            dx = a.x - ego.x
            dy = a.y - ego.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            # CBF: h(x) = dist^2 - d_safe^2
            h_val = dist ** 2 - self.d_safe ** 2

            # Relative velocity
            dvx = ego.vx - a.vx
            dvy = ego.vy - a.vy
            h_dot = 2 * (dx * dvx + dy * dvy)

            # CBF condition: h_dot + alpha * h >= 0
            cbf_margin = h_dot + self.alpha * h_val

            ttc = compute_ttc(
                np.array([ego.x, ego.y]), np.array([ego.vx, ego.vy]),
                np.array([a.x, a.y]), np.array([a.vx, a.vy]),
            )
            min_ttc = min(min_ttc, ttc)

            if cbf_margin < 0:
                # CBF constraint violated — exclude region around agent
                # Exclusion radius proportional to violation magnitude
                buffer_radius = max(self.d_safe * 1.2, abs(cbf_margin) ** 0.5 + 2.0)
                exclusion = Point(a.x, a.y).buffer(buffer_radius)
                feasible = feasible.difference(exclusion)
                reasoning.append(f"CBF violation for {a.id}: h={h_val:.1f}, margin={cbf_margin:.1f}")

        from shapely.geometry import MultiPolygon
        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)
        if feasible.is_empty:
            feasible = Point(ego.x, ego.y).buffer(3.0)
            reasoning.append("CBF: all excluded, fallback")

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') else boundary

        # Set mode based on actual safety state (bug fix 260508)
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
            reasoning=reasoning if reasoning else ["CBF: no constraint needed"],
        )



class AblationMethod:
    """Configurable ablation of our full method.

    Disables specific components to measure their individual contribution.
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self._config = config

        from coop_safety.interface import SafetyConstraintModule
        self._module = SafetyConstraintModule(config)

        # Store which components are disabled for logging
        self._disabled = []
        if config.get("disable_risk_graph"):
            self._disabled.append("RiskGraph")
        if config.get("disable_risk_events"):
            self._disabled.append("RiskEvents")
        if config.get("disable_feasibility_check"):
            self._disabled.append("FeasibilityCheck")
        if config.get("disable_min_harm"):
            self._disabled.append("MinHarm")
        if config.get("disable_blind_spot"):
            self._disabled.append("BlindSpot")
        if config.get("risk_map_only"):
            self._disabled.append("RiskGraph+RiskEvents (RiskMap only)")

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Run constrain with ablated components."""
        # For ablations, we modify the pipeline behavior through config flags
        # The actual ablation logic is handled in the module's constrain method
        # by checking config flags

        result = self._module.constrain(perception)

        # Override mode info for ablation tracking
        if self._disabled:
            result.reasoning.insert(0, f"[ABLATION] Disabled: {', '.join(self._disabled)}")

        return result


# ============================================================
# Ablation configurations
# ============================================================

def get_all_methods() -> dict:
    """Return all methods (baselines + ablations + full) for experiment."""
    from coop_safety.interface import SafetyConstraintModule

    methods = {}

    # Baselines
    methods["NoConstraint"] = NoConstraint()
    methods["RSS-Only"] = RSSOnly()
    methods["CBF-Based"] = CBFBased()

    # Our full method
    full = SafetyConstraintModule()
    full.name = "Ours-Full"
    methods["Ours-Full"] = full

    # Ablation: No RiskGraph (skip TTC tightening in step 3.3)
    methods["Ours-NoRiskGraph"] = AblationMethod(
        "Ours-NoRiskGraph",
        {"disable_risk_graph": True, "ttc_warning": 0.0},
    )

    # Ablation: No RiskEvents (skip event-based exclusion in step 3.2)
    methods["Ours-NoRiskEvents"] = AblationMethod(
        "Ours-NoRiskEvents",
        {"disable_risk_events": True, "min_probability": 999},
    )

    # Ablation: No long-term feasibility check (skip step 3.5)
    methods["Ours-NoFeasCheck"] = AblationMethod(
        "Ours-NoFeasCheck",
        {"disable_feasibility_check": True, "check_horizon": 0.0},
    )

    # Ablation: No minimum harm (skip step 3.6)
    methods["Ours-NoMinHarm"] = AblationMethod(
        "Ours-NoMinHarm",
        {"disable_min_harm": True},
    )

    # Ablation: No blind spot inference
    methods["Ours-NoBlindSpot"] = AblationMethod(
        "Ours-NoBlindSpot",
        {"disable_blind_spot": True},
    )

    # Ablation: RiskMap only (no RiskGraph, no RiskEvents)
    methods["Ours-RiskMapOnly"] = AblationMethod(
        "Ours-RiskMapOnly",
        {"risk_map_only": True, "ttc_warning": 0.0, "min_probability": 999},
    )

    return methods

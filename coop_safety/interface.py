from __future__ import annotations
"""Safety Constraint Interface — the core API of the safety module.

This module defines the independent safety constraint interface that takes
cooperative perception results as input and outputs constrained safe action spaces.
It is designed to be pluggable into any autonomous driving algorithm (end-to-end or modular).

Architecture:
    Cooperative Perception → SafetyConstraintModule → Safe Action Space
                                    │
                     ┌──────────────┼──────────────────┐
                     ▼              ▼                   ▼
                 RiskMap       RiskGraph           RiskEvents
                     │              │                   │
                     ▼              ▼                   ▼
              FeasibleRegion → HierarchicalConstraint → FeasibilityCheck
                                                        │
                                                   MinHarmMode (fallback)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# =============================================================================
# Input Data Structures
# =============================================================================

@dataclass
class VehicleState:
    """State of a single vehicle."""
    id: str
    x: float                    # Position x (meters, global frame)
    y: float                    # Position y (meters, global frame)
    heading: float              # Heading angle (radians)
    velocity: float             # Speed (m/s)
    vx: float = 0.0            # Velocity x component
    vy: float = 0.0            # Velocity y component
    acceleration: float = 0.0  # Current acceleration (m/s^2)
    yaw_rate: float = 0.0      # Yaw rate (rad/s)
    length: float = 4.5        # Vehicle length (meters)
    width: float = 1.8         # Vehicle width (meters)
    mass: float = 1500.0       # Vehicle mass (kg)
    vehicle_type: str = "car"  # "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"


class AgentType(Enum):
    """Type of traffic participant."""
    EGO = "ego"
    CAV = "cav"             # Connected Automated Vehicle (cooperative)
    VEHICLE = "vehicle"     # Non-connected vehicle
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    UNKNOWN = "unknown"


@dataclass
class Agent:
    """A traffic participant with perception metadata."""
    state: VehicleState
    agent_type: AgentType = AgentType.VEHICLE
    is_visible: bool = True           # Whether directly observed (vs. blind-spot inferred)
    confidence: float = 1.0           # Detection confidence [0, 1]
    predicted_trajectory: Optional[np.ndarray] = None  # (T, 2) future positions
    source: str = "ego"               # Which sensor/vehicle observed this agent


@dataclass
class LaneInfo:
    """Lane topology information."""
    lane_id: str
    center_line: np.ndarray       # (N, 2) polyline
    left_boundary: np.ndarray     # (N, 2) polyline
    right_boundary: np.ndarray    # (N, 2) polyline
    speed_limit: float = 30.0     # m/s
    lane_type: str = "driving"    # "driving", "shoulder", "bike", "sidewalk"


@dataclass
class BlindSpot:
    """A region not directly observable (occluded area)."""
    polygon: np.ndarray           # (N, 2) vertices of the blind-spot polygon
    occluder_id: Optional[str] = None  # ID of the occluding object


@dataclass
class PerceptionResult:
    """Complete cooperative perception output — input to SafetyConstraintModule.

    This is the unified input interface. All cooperative perception systems
    should convert their output to this format.
    """
    timestamp: float                          # Current time (seconds)
    ego: VehicleState                         # Ego vehicle state
    agents: list[Agent] = field(default_factory=list)  # All detected agents
    lanes: list[LaneInfo] = field(default_factory=list)  # Lane information
    blind_spots: list[BlindSpot] = field(default_factory=list)  # Occluded regions
    visibility_map: Optional[np.ndarray] = None  # (H, W) occupancy grid, 1=visible 0=occluded


# =============================================================================
# Risk Assessment Output Structures
# =============================================================================

class RiskLevel(Enum):
    """Risk level classification."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class RiskRegion:
    """A spatial region with associated risk level (RiskMap output)."""
    polygon: np.ndarray        # (N, 2) vertices
    risk_level: RiskLevel
    risk_score: float          # Continuous risk value [0, 1]
    density: float = 0.0      # Traffic density in the region
    uncontrolled_agents: int = 0  # Number of uncontrolled agents (pedestrians, cyclists)


@dataclass
class ConflictEdge:
    """A pairwise conflict relationship (RiskGraph output)."""
    agent_a_id: str
    agent_b_id: str
    ttc: float                 # Time-to-Collision (seconds), inf if no collision
    collision_probability: float  # Estimated collision probability [0, 1]
    collision_point: Optional[np.ndarray] = None  # (2,) predicted collision location
    time_window: tuple[float, float] = (0.0, 0.0)  # (start, end) of conflict window


@dataclass
class CollisionEvent:
    """A specific potential collision scenario (RiskEvents output)."""
    event_id: str
    participants: list[str]    # Agent IDs involved
    spatial_region: np.ndarray  # (N, 2) polygon where collision may occur
    time_window: tuple[float, float]  # (start, end) seconds from now
    collision_type: str        # "rear_end", "side", "head_on", "pedestrian", "intersection"
    severity: float            # Severity score [0, 1] (based on speed, mass, angle)
    probability: float         # Event probability [0, 1]


@dataclass
class ThreeLayerRisk:
    """Combined three-layer risk assessment output."""
    risk_map: list[RiskRegion]
    risk_graph: list[ConflictEdge]
    risk_events: list[CollisionEvent]
    timestamp: float


# =============================================================================
# Safety Constraint Output Structures
# =============================================================================

class ConstraintMode(Enum):
    """Current operating mode of the safety constraint."""
    NORMAL = "normal"              # Normal operation with safety margins
    CONSERVATIVE = "conservative"  # Tightened constraints due to high risk
    MINIMUM_HARM = "minimum_harm"  # Unavoidable collision, minimizing damage


@dataclass
class SafeActionSpace:
    """The constrained safe action space — output of SafetyConstraintModule.

    This is the core output that downstream planners should respect.
    Any action within this space is considered safe.
    """
    # Spatial constraint
    feasible_region: np.ndarray      # (N, 2) polygon vertices of safe drivable area

    # Kinematic constraints
    max_acceleration: float          # Maximum safe acceleration (m/s^2)
    min_acceleration: float          # Maximum braking (m/s^2, negative)
    max_steering: float              # Maximum steering angle (radians)

    # Speed constraints
    max_speed: float                 # Maximum safe speed (m/s)
    min_speed: float = 0.0          # Minimum speed (usually 0)

    # Metadata
    mode: ConstraintMode = ConstraintMode.NORMAL
    safety_margin_ttc: float = float('inf')  # Minimum TTC in current constraint
    feasibility_horizon: float = 3.0  # Seconds of future feasibility checked

    # Reasoning chain (for interpretability — Innovation Point 1)
    reasoning: list[str] = field(default_factory=list)
    risk_sources: list[str] = field(default_factory=list)

    # Long-term feasibility check result (Innovation Point 2)
    future_feasible: bool = True
    future_feasible_horizon: float = 0.0

    # Minimum harm info (Innovation Point 3, only when mode == MINIMUM_HARM)
    min_harm_target: Optional[CollisionEvent] = None
    min_harm_action: Optional[np.ndarray] = None  # Recommended action to minimize harm


# =============================================================================
# Main Interface
# =============================================================================

class SafetyConstraintModule:
    """Independent safety constraint interface.

    This is the main entry point. It receives cooperative perception results
    and outputs a safe action space. Designed to be pluggable into any
    autonomous driving algorithm (Innovation Point 4).

    Usage:
        module = SafetyConstraintModule(config)
        perception = get_cooperative_perception()  # From any V2X system
        safe_space = module.constrain(perception)
        action = planner.plan(safe_space)  # Any planner respects the constraint
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        from .risk.risk_map import RiskMapBuilder
        from .risk.risk_graph import RiskGraphBuilder
        from .risk.risk_events import RiskEventEnumerator
        from .constraint.feasible_region import FeasibleRegionComputer
        from .constraint.hierarchical import HierarchicalConstraint
        from .constraint.feasibility_check import FeasibilityChecker
        from .constraint.min_harm import MinimumHarmPlanner
        from .perception.blind_spot import identify_blind_spots, infer_hidden_agents

        self._risk_map_builder = RiskMapBuilder(self.config)
        self._risk_graph_builder = RiskGraphBuilder(self.config)
        self._risk_event_enumerator = RiskEventEnumerator(self.config)
        self._feasible_computer = FeasibleRegionComputer(self.config)
        self._hierarchical = HierarchicalConstraint(self.config)
        self._feasibility_checker = FeasibilityChecker(self.config)
        self._min_harm_planner = MinimumHarmPlanner(self.config)
        self._identify_blind_spots = identify_blind_spots
        self._infer_hidden_agents = infer_hidden_agents

    def constrain(self, perception: PerceptionResult) -> SafeActionSpace:
        """Main API: Perception → Safe Action Space.

        Steps:
            1. Blind spot inference → augment agent list
            2. Three-layer risk assessment (RiskMap → RiskGraph → RiskEvents)
            3. Compute initial feasible region (dynamics + road)
            4. Tighten by RiskEvents (step 3.2)
            5. Tighten by TTC / RiskGraph (step 3.3)
            6. Tighten by risk zones / RiskMap (step 3.4)
            7. Long-term feasibility check (step 3.5)
            8. If failed: relax + retry, or minimum harm (step 3.6)
        """
        from shapely.geometry import Polygon as ShapelyPolygon

        reasoning = []
        ego = perception.ego

        # Step 1: Blind spot inference (Plan 2.2)
        blind_spots = self._identify_blind_spots(perception)
        phantom_agents = self._infer_hidden_agents(blind_spots, ego)
        all_agents = perception.agents + phantom_agents
        if phantom_agents:
            reasoning.append(f"Inferred {len(phantom_agents)} phantom agents in blind spots")

        # Step 2: Three-layer risk assessment (Plan 2.4-2.6)
        risk = self.assess_risk_with_agents(perception, all_agents, blind_spots)

        # Build agent lookup
        agents_by_id = {a.state.id: a for a in all_agents}

        # Step 3: Initial feasible region (Plan 3.1)
        feasible = self._feasible_computer.compute_initial(ego, perception.lanes)
        reasoning.append(f"Initial feasible area: {feasible.area:.1f}m²")

        # Step 4: Tighten by RiskEvents (Plan 3.2) — with directional exclusion
        ego_pos = np.array([ego.x, ego.y])
        feasible = self._feasible_computer.tighten_by_events(feasible, risk.risk_events, ego_pos)

        # Step 5-6: Hierarchical tightening (Plan 3.3-3.4)
        feasible, min_ttc, hier_reasons = self._hierarchical.apply_full_hierarchy(
            feasible, risk.risk_map, risk.risk_graph, ego, agents=all_agents
        )
        reasoning.extend(hier_reasons)

        # Step 7: Long-term feasibility check (Plan 3.5)
        is_feasible, achieved_horizon, feas_reasons = self._feasibility_checker.check(
            feasible, ego, all_agents
        )
        reasoning.extend(feas_reasons)

        # Step 8: If not feasible → relax or minimum harm (Plan 3.6)
        mode = ConstraintMode.NORMAL
        if not is_feasible or feasible.is_empty:
            reasoning.append("Feasibility check FAILED — entering relaxation/min-harm")

            # Try relaxation: re-run with looser thresholds
            relaxed = self._try_relaxation(perception, all_agents, blind_spots, ego)
            if relaxed is not None:
                feasible = relaxed
                mode = ConstraintMode.CONSERVATIVE
                reasoning.append("Relaxed constraints applied")
            else:
                # Minimum harm mode (Innovation Point 3)
                return self._min_harm_planner.plan(ego, risk.risk_events, feasible)

        # Build output
        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') else np.array([[ego.x, ego.y]])
        from .perception.dynamics import get_dynamics_params
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)

        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=mode,
            safety_margin_ttc=min_ttc,
            feasibility_horizon=achieved_horizon,
            reasoning=reasoning,
            risk_sources=[e.event_id for e in risk.risk_events[:5]],
            future_feasible=is_feasible,
            future_feasible_horizon=achieved_horizon,
        )

    def assess_risk(self, perception: PerceptionResult) -> ThreeLayerRisk:
        """Compute three-layer risk assessment."""
        from .perception.blind_spot import identify_blind_spots, infer_hidden_agents
        blind_spots = identify_blind_spots(perception)
        phantom_agents = infer_hidden_agents(blind_spots, perception.ego)
        all_agents = perception.agents + phantom_agents
        return self.assess_risk_with_agents(perception, all_agents, blind_spots)

    def assess_risk_with_agents(self, perception: PerceptionResult,
                                all_agents: list, blind_spots: list) -> ThreeLayerRisk:
        """Internal: risk assessment with augmented agent list."""
        bs_polys = [bs.polygon for bs in blind_spots]
        risk_map = self._risk_map_builder.build(perception, bs_polys)

        risk_graph = self._risk_graph_builder.build(perception.ego, all_agents)

        agents_by_id = {a.state.id: a for a in all_agents}
        risk_events = self._risk_event_enumerator.enumerate(
            risk_graph, agents_by_id, perception.ego
        )

        return ThreeLayerRisk(
            risk_map=risk_map,
            risk_graph=risk_graph,
            risk_events=risk_events,
            timestamp=perception.timestamp,
        )

    def _try_relaxation(self, perception, all_agents, blind_spots, ego):
        """Try relaxing constraints progressively. Returns None quickly if infeasible."""
        # Only try ONE relaxation level (not 3) for speed
        relaxed_config = dict(self.config)
        relaxed_config["high_risk_exclude_ratio"] = 0.3
        relaxed_config["medium_risk_exclude_ratio"] = 0.0
        relaxed_config["ttc_warning"] = 5.0

        from .constraint.hierarchical import HierarchicalConstraint
        relaxed_hier = HierarchicalConstraint(relaxed_config)
        feasible = self._feasible_computer.compute_initial(ego, perception.lanes)

        risk = self.assess_risk_with_agents(perception, all_agents, blind_spots)
        feasible = self._feasible_computer.tighten_by_events(feasible, risk.risk_events)
        feasible, _, _ = relaxed_hier.apply_full_hierarchy(
            feasible, risk.risk_map, risk.risk_graph, ego
        )

        if not feasible.is_empty and feasible.area > 5.0:
            return feasible
        return None

    def reset(self):
        """Reset internal state (call between episodes)."""
        pass

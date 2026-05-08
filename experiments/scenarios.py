"""Reproducible scenario generator for safety constraint evaluation.

Generates synthetic cooperative driving scenarios with deterministic seeds.
Each scenario defines ego vehicle, surrounding agents, lanes, and blind spots.

IMPORTANT: All scenarios use fixed random seeds for full reproducibility.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    LaneInfo, BlindSpot,
)


@dataclass
class ScenarioConfig:
    """Configuration for a scenario."""
    name: str
    description: str
    seed: int
    # Expected difficulty for the safety module
    difficulty: str  # "easy", "medium", "hard", "critical"


def _make_lane(lane_id: str, y_center: float, width: float = 3.5,
               x_start: float = -100, x_end: float = 200) -> LaneInfo:
    """Create a straight lane along the x-axis."""
    n_points = 50
    xs = np.linspace(x_start, x_end, n_points)
    half_w = width / 2
    return LaneInfo(
        lane_id=lane_id,
        center_line=np.column_stack([xs, np.full(n_points, y_center)]),
        left_boundary=np.column_stack([xs, np.full(n_points, y_center + half_w)]),
        right_boundary=np.column_stack([xs, np.full(n_points, y_center - half_w)]),
        speed_limit=30.0,
        lane_type="driving",
    )


def scenario_highway_normal(seed: int = 42) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 1: Normal highway driving — sparse traffic, no conflict.

    Ego driving at 25 m/s. A few vehicles in adjacent lanes, all well separated.
    Expected: NORMAL mode, large feasible region, no constraint tightening.
    """
    rng = np.random.RandomState(seed)

    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=25.0, vx=25.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        Agent(state=VehicleState(
            id="v1", x=50.0, y=0.0, heading=0.0,
            velocity=23.0, vx=23.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        Agent(state=VehicleState(
            id="v2", x=30.0, y=3.5, heading=0.0,
            velocity=26.0, vx=26.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        Agent(state=VehicleState(
            id="v3", x=-40.0, y=-3.5, heading=0.0,
            velocity=24.0, vx=24.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
    ]

    lanes = [_make_lane("lane_0", 0.0), _make_lane("lane_1", 3.5), _make_lane("lane_2", -3.5)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="highway_normal", description="Normal highway, sparse traffic",
        seed=seed, difficulty="easy",
    )


def scenario_highway_dense(seed: int = 43) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 2: Dense highway traffic — multiple vehicles, moderate risk.

    Ego in middle lane. Vehicles ahead braking, vehicles on both sides.
    Expected: some TTC tightening, feasible but constrained.
    """
    rng = np.random.RandomState(seed)

    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=22.0, vx=22.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        # Slow vehicle ahead (braking)
        Agent(state=VehicleState(
            id="v_ahead", x=25.0, y=0.0, heading=0.0,
            velocity=15.0, vx=15.0, vy=0.0,
            acceleration=-2.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Vehicle in left lane
        Agent(state=VehicleState(
            id="v_left", x=10.0, y=3.5, heading=0.0,
            velocity=21.0, vx=21.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Vehicle in right lane
        Agent(state=VehicleState(
            id="v_right", x=5.0, y=-3.5, heading=0.0,
            velocity=20.0, vx=20.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Vehicle behind
        Agent(state=VehicleState(
            id="v_behind", x=-15.0, y=0.0, heading=0.0,
            velocity=25.0, vx=25.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
    ]

    lanes = [_make_lane("lane_0", 0.0), _make_lane("lane_1", 3.5), _make_lane("lane_2", -3.5)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="highway_dense", description="Dense highway, braking vehicle ahead",
        seed=seed, difficulty="medium",
    )


def scenario_intersection_conflict(seed: int = 44) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 3: Intersection with cross traffic — high conflict risk.

    Ego approaching intersection. Cross traffic from the left.
    Expected: high risk events, significant tightening, possibly CONSERVATIVE.
    """
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=12.0, vx=12.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        # Cross traffic from left
        Agent(state=VehicleState(
            id="cross_1", x=15.0, y=25.0, heading=-np.pi / 2,
            velocity=10.0, vx=0.0, vy=-10.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Another cross vehicle
        Agent(state=VehicleState(
            id="cross_2", x=20.0, y=30.0, heading=-np.pi / 2,
            velocity=11.0, vx=0.0, vy=-11.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Pedestrian crossing
        Agent(state=VehicleState(
            id="ped_1", x=18.0, y=-3.0, heading=np.pi / 2,
            velocity=1.5, vx=0.0, vy=1.5,
            length=0.5, width=0.5, mass=70,
            vehicle_type="pedestrian",
        ), agent_type=AgentType.PEDESTRIAN),
    ]

    lanes = [_make_lane("ego_lane", 0.0)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="intersection_conflict", description="Intersection with cross traffic and pedestrian",
        seed=seed, difficulty="hard",
    )


def scenario_blind_spot_danger(seed: int = 45) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 4: Blind spot with hidden vehicle — cooperative perception needed.

    Large truck in adjacent lane creates blind spot. A hidden vehicle may be behind it.
    Tests blind spot inference and conservative constraint.
    """
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=20.0, vx=20.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        # Large truck creating blind spot
        Agent(state=VehicleState(
            id="truck", x=15.0, y=3.5, heading=0.0,
            velocity=18.0, vx=18.0, vy=0.0,
            length=12.0, width=2.5, mass=10000,
            vehicle_type="truck",
        ), agent_type=AgentType.VEHICLE),
        # Vehicle that would be hidden (marked as low confidence from cooperative perception)
        Agent(state=VehicleState(
            id="hidden_v", x=25.0, y=3.5, heading=0.0,
            velocity=19.0, vx=19.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE, is_visible=False, confidence=0.4,
              source="cooperative_v2v"),
    ]

    lanes = [_make_lane("lane_0", 0.0), _make_lane("lane_1", 3.5)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="blind_spot_danger", description="Truck blind spot with hidden vehicle (cooperative detection)",
        seed=seed, difficulty="medium",
    )


def scenario_head_on_unavoidable(seed: int = 46) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 5: Near head-on collision — tests minimum harm mode.

    Oncoming vehicle in ego's lane at close range. Very limited time to react.
    Expected: MINIMUM_HARM mode triggered.
    """
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=20.0, vx=20.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        # Oncoming vehicle in same lane
        Agent(state=VehicleState(
            id="oncoming", x=30.0, y=0.0, heading=np.pi,
            velocity=20.0, vx=-20.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        # Vehicles blocking escape routes
        Agent(state=VehicleState(
            id="block_left", x=5.0, y=3.5, heading=0.0,
            velocity=20.0, vx=20.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
        Agent(state=VehicleState(
            id="block_right", x=5.0, y=-3.5, heading=0.0,
            velocity=20.0, vx=20.0, vy=0.0,
            length=4.5, width=1.8, mass=1500,
        ), agent_type=AgentType.VEHICLE),
    ]

    lanes = [_make_lane("lane_0", 0.0), _make_lane("lane_1", 3.5), _make_lane("lane_2", -3.5)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="head_on_unavoidable",
        description="Near head-on with blocked escape routes — minimum harm test",
        seed=seed, difficulty="critical",
    )


def scenario_pedestrian_crossing(seed: int = 47) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 6: Multiple pedestrians crossing — uncontrolled agents.

    Urban scenario with pedestrians crossing at various points.
    Tests risk map's uncontrolled agent scoring.
    """
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=8.0, vx=8.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    agents = [
        Agent(state=VehicleState(
            id="ped_1", x=12.0, y=-4.0, heading=np.pi / 2,
            velocity=1.2, vx=0.0, vy=1.2,
            length=0.5, width=0.5, mass=70, vehicle_type="pedestrian",
        ), agent_type=AgentType.PEDESTRIAN),
        Agent(state=VehicleState(
            id="ped_2", x=15.0, y=5.0, heading=-np.pi / 2,
            velocity=1.0, vx=0.0, vy=-1.0,
            length=0.5, width=0.5, mass=65, vehicle_type="pedestrian",
        ), agent_type=AgentType.PEDESTRIAN),
        Agent(state=VehicleState(
            id="cyclist", x=20.0, y=-2.0, heading=np.pi / 4,
            velocity=4.0, vx=2.83, vy=2.83,
            length=1.8, width=0.6, mass=90, vehicle_type="bicycle",
        ), agent_type=AgentType.CYCLIST),
    ]

    lanes = [_make_lane("lane_0", 0.0, width=4.0)]

    return PerceptionResult(
        timestamp=0.0, ego=ego, agents=agents, lanes=lanes,
    ), ScenarioConfig(
        name="pedestrian_crossing",
        description="Urban scenario with pedestrians and cyclist",
        seed=seed, difficulty="hard",
    )


def scenario_cooperative_advantage(seed: int = 48) -> tuple[PerceptionResult, ScenarioConfig]:
    """Scenario 7: Cooperative perception reveals hidden threat.

    Without V2X: ego can't see vehicle behind truck.
    With V2X: cooperative partner detects it, enabling earlier constraint.
    Run twice (with/without cooperative info) to show V2X benefit.
    """
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=18.0, vx=18.0, vy=0.0,
        length=4.5, width=1.8, mass=1500,
    )

    # Truck occluder
    truck = Agent(state=VehicleState(
        id="truck", x=20.0, y=3.5, heading=0.0,
        velocity=15.0, vx=15.0, vy=0.0,
        length=12.0, width=2.5, mass=10000, vehicle_type="truck",
    ), agent_type=AgentType.VEHICLE)

    # Hidden vehicle merging into ego's lane — detected by cooperative partner
    hidden = Agent(state=VehicleState(
        id="merger", x=35.0, y=3.5, heading=-0.3,
        velocity=16.0, vx=15.5, vy=-4.7,
        length=4.5, width=1.8, mass=1500,
    ), agent_type=AgentType.CAV, is_visible=True, confidence=0.85,
          source="cooperative_cav_2")

    lanes = [_make_lane("lane_0", 0.0), _make_lane("lane_1", 3.5)]

    # Full cooperative version (with hidden vehicle info)
    perception_coop = PerceptionResult(
        timestamp=0.0, ego=ego, agents=[truck, hidden], lanes=lanes,
    )

    return perception_coop, ScenarioConfig(
        name="cooperative_advantage",
        description="Cooperative perception reveals merging vehicle behind truck",
        seed=seed, difficulty="medium",
    )


# Registry of all scenarios
ALL_SCENARIOS = {
    "highway_normal": scenario_highway_normal,
    "highway_dense": scenario_highway_dense,
    "intersection_conflict": scenario_intersection_conflict,
    "blind_spot_danger": scenario_blind_spot_danger,
    "head_on_unavoidable": scenario_head_on_unavoidable,
    "pedestrian_crossing": scenario_pedestrian_crossing,
    "cooperative_advantage": scenario_cooperative_advantage,
}


def get_all_scenarios() -> list[tuple[PerceptionResult, ScenarioConfig]]:
    """Get all scenarios with their configs."""
    return [fn() for fn in ALL_SCENARIOS.values()]

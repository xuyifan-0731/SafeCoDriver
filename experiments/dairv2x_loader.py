"""DAIR-V2X dataset loader — converts real-world cooperative perception data
to our PerceptionResult format for safety constraint evaluation.

DAIR-V2X (CVPR 2022) is a real-world Vehicle-Infrastructure Cooperative dataset
with LiDAR point clouds and 3D object annotations from both vehicle and
infrastructure sides.

Data path: /raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Example/
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType, LaneInfo,
)


# Type mapping from DAIR-V2X to our format
DAIR_TYPE_MAP = {
    "car": ("car", AgentType.VEHICLE),
    "van": ("car", AgentType.VEHICLE),
    "truck": ("truck", AgentType.VEHICLE),
    "bus": ("bus", AgentType.VEHICLE),
    "pedestrian": ("pedestrian", AgentType.PEDESTRIAN),
    "cyclist": ("bicycle", AgentType.CYCLIST),
    "motorcyclist": ("motorcycle", AgentType.VEHICLE),
    "trafficcone": ("unknown", AgentType.UNKNOWN),
    "barrowlist": ("unknown", AgentType.UNKNOWN),
}

# Default mass by type (kg)
DEFAULT_MASS = {
    "car": 1500, "van": 2000, "truck": 8000, "bus": 12000,
    "pedestrian": 70, "cyclist": 80, "motorcyclist": 200,
}


class DAIRv2xLoader:
    """Load DAIR-V2X-C cooperative data and convert to PerceptionResult.

    The dataset provides world-coordinate 3D labels from both vehicle
    and infrastructure sensors — this simulates cooperative perception.
    """

    def __init__(self, data_dir: str, ego_speed: float = 10.0):
        """
        Args:
            data_dir: Path to the cooperative data directory
                      (e.g., .../example-cooperative-vehicle-infrastructure/)
            ego_speed: Assumed ego vehicle speed (m/s) since DAIR-V2X
                       doesn't provide ego velocity in labels. In a full
                       pipeline this would come from odometry.
        """
        self.data_dir = Path(data_dir)
        self.coop_label_dir = self.data_dir / "cooperative" / "label_world"
        self.ego_speed = ego_speed

        # Get sorted frame list
        self.frames = sorted([
            f.stem for f in self.coop_label_dir.glob("*.json")
        ])
        print(f"[DAIRv2xLoader] Loaded {len(self.frames)} frames from {data_dir}")

    def __len__(self):
        return len(self.frames)

    def load_frame(self, idx: int) -> PerceptionResult:
        """Load one frame and convert to PerceptionResult.

        Args:
            idx: Frame index

        Returns:
            PerceptionResult with ego + agents from cooperative labels
        """
        frame_id = self.frames[idx]
        label_path = self.coop_label_dir / f"{frame_id}.json"

        with open(label_path) as f:
            labels = json.load(f)

        # In DAIR-V2X, the first detected object closest to the sensor origin
        # is approximately the ego vehicle's surroundings. We pick the first
        # car-type object as a reference point and treat others as agents.
        # (In a real system, ego state comes from localization, not detection.)

        # Find all objects and sort by distance to approximate ego position
        objects = []
        for obj in labels:
            loc = obj["3d_location"]
            dim = obj["3d_dimensions"]
            obj_type = obj.get("type", "car").lower()

            # Skip unknown small objects
            if obj_type in ("trafficcone", "barrowlist"):
                continue

            objects.append({
                "type": obj_type,
                "x": loc["x"],
                "y": loc["y"],
                "z": loc["z"],
                "length": dim["l"],
                "width": dim["w"],
                "height": dim["h"],
                "rotation": obj.get("rotation", obj.get("alpha", 0.0)),
                "occluded": obj.get("occluded_state", 0),
            })

        if not objects:
            # Empty frame — return default
            return PerceptionResult(
                timestamp=float(idx),
                ego=VehicleState(id="ego", x=0, y=0, heading=0, velocity=0),
            )

        # Use centroid of all vehicles as rough scene center
        vehicle_objs = [o for o in objects if o["type"] in ("car", "van")]
        if vehicle_objs:
            cx = np.mean([o["x"] for o in vehicle_objs])
            cy = np.mean([o["y"] for o in vehicle_objs])
        else:
            cx = objects[0]["x"]
            cy = objects[0]["y"]

        # Pick the vehicle closest to scene center as "ego"
        # (In real use, ego state would come from localization)
        ego_candidates = [o for o in objects if o["type"] in ("car", "van")]
        if not ego_candidates:
            ego_candidates = objects

        ego_obj = min(ego_candidates, key=lambda o: (o["x"] - cx) ** 2 + (o["y"] - cy) ** 2)
        objects.remove(ego_obj)

        # Build ego state
        ego_heading = float(ego_obj["rotation"])
        ego = VehicleState(
            id="ego",
            x=ego_obj["x"],
            y=ego_obj["y"],
            heading=ego_heading,
            velocity=self.ego_speed,
            vx=self.ego_speed * np.cos(ego_heading),
            vy=self.ego_speed * np.sin(ego_heading),
            length=ego_obj["length"],
            width=ego_obj["width"],
            mass=DEFAULT_MASS.get(ego_obj["type"], 1500),
            vehicle_type="car",
        )

        # Build agents
        agents = []
        for i, obj in enumerate(objects):
            our_type, agent_type = DAIR_TYPE_MAP.get(
                obj["type"], ("car", AgentType.VEHICLE)
            )
            heading = float(obj["rotation"])
            # Estimate velocity: assume all vehicles move forward at similar speed
            # (DAIR-V2X labels are single-frame, no velocity info)
            if our_type in ("pedestrian",):
                speed = 1.5
            elif our_type in ("bicycle",):
                speed = 4.0
            else:
                # Slight random variation based on position for diversity
                speed = self.ego_speed + (hash(f"{frame_id}_{i}") % 7 - 3)
                speed = max(speed, 0)

            agent = Agent(
                state=VehicleState(
                    id=f"obj_{i:03d}",
                    x=obj["x"],
                    y=obj["y"],
                    heading=heading,
                    velocity=speed,
                    vx=speed * np.cos(heading),
                    vy=speed * np.sin(heading),
                    length=obj["length"],
                    width=obj["width"],
                    mass=DEFAULT_MASS.get(obj["type"], 1500),
                    vehicle_type=our_type,
                ),
                agent_type=agent_type,
                is_visible=obj["occluded"] == 0,
                confidence=0.9 if obj["occluded"] == 0 else 0.6,
                source="cooperative_v2i" if obj["occluded"] == 0 else "infrastructure_only",
            )
            agents.append(agent)

        # No lane information in DAIR-V2X labels — generate synthetic straight lane
        # along ego heading (sufficient for constraint evaluation)
        lanes = self._generate_lanes(ego)

        return PerceptionResult(
            timestamp=float(idx),
            ego=ego,
            agents=agents,
            lanes=lanes,
        )

    def load_all_frames(self) -> list[PerceptionResult]:
        """Load all frames."""
        return [self.load_frame(i) for i in range(len(self.frames))]

    def _generate_lanes(self, ego: VehicleState, n_lanes: int = 3) -> list[LaneInfo]:
        """Generate synthetic lanes along ego heading direction."""
        lanes = []
        cos_h = np.cos(ego.heading)
        sin_h = np.sin(ego.heading)

        for lane_idx in range(n_lanes):
            lateral_offset = (lane_idx - n_lanes // 2) * 3.5  # 3.5m lane width

            # Lane centerline in global frame
            n_pts = 30
            s = np.linspace(-50, 150, n_pts)
            center = np.column_stack([
                ego.x + s * cos_h - lateral_offset * sin_h,
                ego.y + s * sin_h + lateral_offset * cos_h,
            ])
            left = np.column_stack([
                center[:, 0] - 1.75 * sin_h,
                center[:, 1] + 1.75 * cos_h,
            ])
            right = np.column_stack([
                center[:, 0] + 1.75 * sin_h,
                center[:, 1] - 1.75 * cos_h,
            ])

            lanes.append(LaneInfo(
                lane_id=f"lane_{lane_idx}",
                center_line=center,
                left_boundary=left,
                right_boundary=right,
                speed_limit=15.0,  # Urban speed
                lane_type="driving",
            ))

        return lanes


def get_dairv2x_scenarios(data_dir: Optional[str] = None,
                          n_frames: int = 10,
                          seed: int = 42) -> list[tuple]:
    """Get DAIR-V2X frames as experiment scenarios.

    Args:
        data_dir: Path to cooperative data. If None, uses default example path.
        n_frames: Number of frames to sample (evenly spaced)
        seed: Random seed for frame selection

    Returns:
        List of (PerceptionResult, ScenarioConfig) tuples
    """
    from experiments.scenarios import ScenarioConfig

    if data_dir is None:
        data_dir = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Example/example-cooperative-vehicle-infrastructure"

    loader = DAIRv2xLoader(data_dir, ego_speed=10.0)

    if len(loader) == 0:
        print("[WARNING] No frames found in DAIR-V2X data")
        return []

    # Evenly sample frames
    rng = np.random.RandomState(seed)
    indices = np.linspace(0, len(loader) - 1, min(n_frames, len(loader)), dtype=int)

    scenarios = []
    for i, idx in enumerate(indices):
        perception = loader.load_frame(idx)
        n_agents = len(perception.agents)

        # Estimate difficulty by agent count
        if n_agents <= 5:
            difficulty = "easy"
        elif n_agents <= 15:
            difficulty = "medium"
        else:
            difficulty = "hard"

        config = ScenarioConfig(
            name=f"dairv2x_frame_{loader.frames[idx]}",
            description=f"DAIR-V2X real-world frame, {n_agents} agents",
            seed=seed + i,
            difficulty=difficulty,
        )
        scenarios.append((perception, config))

    return scenarios

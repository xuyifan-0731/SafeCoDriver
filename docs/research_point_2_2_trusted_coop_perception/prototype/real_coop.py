"""Utilities for DeepAccident real other-vehicle cooperative labels."""
from __future__ import annotations

import copy
import math
import pickle
from pathlib import Path

import numpy as np

from coop_safety.interface import Agent, PerceptionResult, VehicleState
from experiments.deepaccident_loader import TYPE_MAP, parse_label


def _load_calib(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def _calib_path(label_path: Path, role: str) -> Path:
    return Path(str(label_path).replace(f"/{role}/label/", f"/{role}/calib/")).with_suffix(".pkl")


def _transform_point(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    point = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    return transform @ point


def _transform_vector(transform: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    vec = np.array([xyz[0], xyz[1], xyz[2], 0.0], dtype=float)
    return transform @ vec


def _normalize_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def source_lidar_to_target_lidar_transform(source_calib: dict, target_calib: dict) -> np.ndarray:
    """Return transform from source label/lidar frame into target label/lidar frame."""
    source_ego_to_world = source_calib["ego_to_world"]
    target_ego_to_world = target_calib["ego_to_world"]
    source_lidar_to_ego = source_calib["lidar_to_ego"]
    target_lidar_to_ego = target_calib["lidar_to_ego"]
    return (
        np.linalg.inv(target_lidar_to_ego)
        @ np.linalg.inv(target_ego_to_world)
        @ source_ego_to_world
        @ source_lidar_to_ego
    )


def load_aligned_source_message(
    loader,
    scenario_index: int,
    frame_index: int,
    source_role: str = "other_vehicle",
    target_role: str = "ego_vehicle",
    target_ego_exclusion_radius: float = 2.5,
) -> PerceptionResult | None:
    """Load a source role's labels and align them into ego_vehicle label/lidar frame.

    DeepAccident labels are in each vehicle's label/lidar frame. The calibration
    pickles provide `lidar_to_ego` and `ego_to_world`; the correct alignment is:

    source label/lidar -> source ego -> world -> target ego -> target label/lidar.
    """
    scenario = loader.scenarios[scenario_index]
    fname = scenario["frames"][frame_index]
    ego_label_path = Path(scenario["ego_dir"]) / fname
    source_label_path = Path(str(ego_label_path).replace(f"/{target_role}/label/", f"/{source_role}/label/"))
    if not source_label_path.exists():
        return None

    ego_label = parse_label(str(ego_label_path), include_invisible=True)
    source_label = parse_label(str(source_label_path), include_invisible=True)
    ego_visible_by_id = {a["id"]: bool(a["visible"]) for a in ego_label["agents"] if a["id"] != -100}

    target_calib = _load_calib(_calib_path(ego_label_path, target_role))
    source_calib = _load_calib(_calib_path(source_label_path, source_role))
    transform = source_lidar_to_target_lidar_transform(source_calib, target_calib)

    ego = VehicleState(
        id="ego",
        x=0.0,
        y=0.0,
        heading=0.0,
        velocity=ego_label["ego_speed"],
        vx=ego_label["ego_speed"],
        vy=0.0,
        yaw_rate=ego_label["ego_yaw_rate"],
        length=4.5,
        width=1.8,
    )

    agents = []
    for obj in source_label["agents"]:
        if obj["id"] == -100:
            continue
        our_type, agent_type = TYPE_MAP.get(obj["type"], ("car", None))
        if agent_type is None:
            continue

        aligned = _transform_point(transform, np.array([obj["x"], obj["y"], obj["z"]], dtype=float))
        if math.hypot(float(aligned[0]), float(aligned[1])) < target_ego_exclusion_radius:
            continue
        vel = _transform_vector(transform, np.array([obj["vx"], obj["vy"], 0.0], dtype=float))
        heading_vec = _transform_vector(
            transform,
            np.array([math.cos(obj["yaw"]), math.sin(obj["yaw"]), 0.0], dtype=float),
        )
        vx = float(vel[0])
        vy = float(vel[1])
        speed = math.hypot(vx, vy)
        ego_visible = ego_visible_by_id.get(obj["id"], False)
        sender_visible = bool(obj["visible"])
        agents.append(
            Agent(
                state=VehicleState(
                    id=f"obj_{obj['id']}",
                    x=float(aligned[0]),
                    y=float(aligned[1]),
                    heading=_normalize_angle(math.atan2(float(heading_vec[1]), float(heading_vec[0]))),
                    velocity=speed,
                    vx=vx,
                    vy=vy,
                    length=float(obj["length"]),
                    width=float(obj["width"]),
                    vehicle_type=our_type,
                ),
                agent_type=agent_type,
                is_visible=ego_visible,
                confidence=1.0 if sender_visible else 0.2,
                source=f"{source_role}_visible" if sender_visible else f"{source_role}_invisible",
            )
        )

    return PerceptionResult(
        timestamp=0,
        ego=ego,
        agents=agents,
        lanes=[],
        blind_spots=[],
        visibility_map=None,
    )


def load_aligned_other_vehicle_message(loader, scenario_index: int, frame_index: int) -> PerceptionResult | None:
    """Load other_vehicle labels and align them into ego_vehicle label/lidar frame."""
    return load_aligned_source_message(loader, scenario_index, frame_index, source_role="other_vehicle")


def merge_ego_visible_with_real_other(ego_perception: PerceptionResult, real_other: PerceptionResult) -> PerceptionResult:
    """Use ego visible objects and aligned real other-vehicle ego-invisible objects."""
    agents = []
    for agent in ego_perception.agents:
        if agent.is_visible:
            a = copy.deepcopy(agent)
            a.source = "ego"
            a.confidence = 1.0
            agents.append(a)

    ego_visible_ids = {str(agent.state.id) for agent in ego_perception.agents if agent.is_visible}
    for agent in real_other.agents:
        if agent.is_visible:
            continue
        if not str(agent.source).endswith("_visible"):
            continue
        if str(agent.state.id) in ego_visible_ids:
            continue
        a = copy.deepcopy(agent)
        a.source = "coop"
        a.confidence = max(float(a.confidence), 0.8)
        agents.append(a)

    return PerceptionResult(
        timestamp=ego_perception.timestamp,
        ego=copy.deepcopy(ego_perception.ego),
        agents=agents,
        lanes=copy.deepcopy(ego_perception.lanes),
        blind_spots=copy.deepcopy(ego_perception.blind_spots),
        visibility_map=copy.deepcopy(ego_perception.visibility_map),
    )


def filter_sender_visible(real_other: PerceptionResult) -> PerceptionResult:
    """Keep only objects that the other vehicle directly observed."""
    msg = copy.deepcopy(real_other)
    msg.agents = [agent for agent in msg.agents if str(agent.source).endswith("_visible")]
    return msg

"""DAIR-V2X data loader v2 — with REAL velocities from consecutive frames.

Computes ego velocity from novatel_to_world calibration between consecutive frames.
Computes agent velocities by matching objects across consecutive cooperative labels.

This replaces the fake velocity estimation in dairv2x_loader.py.
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


DAIR_TYPE_MAP = {
    "car": ("car", AgentType.VEHICLE),
    "van": ("car", AgentType.VEHICLE),
    "truck": ("truck", AgentType.VEHICLE),
    "bus": ("bus", AgentType.VEHICLE),
    "pedestrian": ("pedestrian", AgentType.PEDESTRIAN),
    "cyclist": ("bicycle", AgentType.CYCLIST),
    "motorcyclist": ("motorcycle", AgentType.VEHICLE),
}

DEFAULT_MASS = {
    "car": 1500, "van": 2000, "truck": 8000, "bus": 12000,
    "pedestrian": 70, "cyclist": 80, "motorcyclist": 200,
}


class DAIRv2xLoaderV2:
    """DAIR-V2X loader with real velocity computation from consecutive frames."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.coop_info = json.load(open(self.data_dir / "cooperative" / "data_info.json"))
        self.veh_info_list = json.load(open(self.data_dir / "vehicle-side" / "data_info.json"))

        # Build vehicle frame lookup: frame_id → {timestamp, batch_id, novatel_path}
        self.veh_lookup = {}
        for entry in self.veh_info_list:
            fid = entry["pointcloud_path"].split("/")[-1].replace(".pcd", "")
            self.veh_lookup[fid] = {
                "timestamp_us": int(entry["pointcloud_timestamp"]),
                "batch_id": entry["batch_id"],
                "novatel_path": entry.get("calib_novatel_to_world_path", ""),
            }

        # Build cooperative frame list with vehicle frame IDs
        self.frames = []  # list of (coop_label_path, vehicle_frame_id)
        for entry in self.coop_info:
            vid = entry["vehicle_pointcloud_path"].split("/")[-1].replace(".pcd", "")
            label_path = entry["cooperative_label_path"]
            self.frames.append((label_path, vid))

        # Group frames by batch for consecutive-frame velocity
        self._build_batch_index()

        print(f"[DAIRv2xLoaderV2] {len(self.frames)} frames, "
              f"{len(self.batches)} batches, "
              f"{self.n_pairs} consecutive pairs ({self.n_pairs/len(self.frames)*100:.0f}%)")

    def _build_batch_index(self):
        """Group frames into batches and sort by timestamp for velocity computation."""
        batch_frames = {}  # batch_id → [(frame_idx, vehicle_fid, timestamp)]
        for idx, (_, vid) in enumerate(self.frames):
            if vid not in self.veh_lookup:
                continue
            info = self.veh_lookup[vid]
            bid = info["batch_id"]
            if bid not in batch_frames:
                batch_frames[bid] = []
            batch_frames[bid].append((idx, vid, info["timestamp_us"]))

        # Sort each batch by timestamp
        self.batches = {}
        self.n_pairs = 0
        self.frame_to_prev = {}  # frame_idx → prev_frame_idx (for velocity)

        for bid, frames in batch_frames.items():
            frames.sort(key=lambda x: x[2])
            self.batches[bid] = frames
            for i in range(1, len(frames)):
                self.frame_to_prev[frames[i][0]] = frames[i - 1][0]
                self.n_pairs += 1

    def __len__(self):
        return len(self.frames)

    def _load_ego_pose(self, vehicle_fid: str) -> Optional[np.ndarray]:
        """Load ego world position from novatel_to_world calibration."""
        info = self.veh_lookup.get(vehicle_fid)
        if not info or not info["novatel_path"]:
            return None
        cal_path = self.data_dir / "vehicle-side" / info["novatel_path"]
        if not cal_path.exists():
            return None
        cal = json.load(open(cal_path))
        t = cal["translation"]
        return np.array([t[0][0], t[1][0], t[2][0]])

    def _load_ego_heading(self, vehicle_fid: str) -> float:
        """Extract ego heading from rotation matrix."""
        info = self.veh_lookup.get(vehicle_fid)
        if not info or not info["novatel_path"]:
            return 0.0
        cal_path = self.data_dir / "vehicle-side" / info["novatel_path"]
        if not cal_path.exists():
            return 0.0
        cal = json.load(open(cal_path))
        R = np.array(cal["rotation"])
        heading = np.arctan2(R[1, 0], R[0, 0])
        return float(heading)

    def _load_labels(self, label_path: str) -> list[dict]:
        """Load cooperative world labels."""
        full_path = self.data_dir / label_path
        if not full_path.exists():
            return []
        labels = json.load(open(full_path))
        objects = []
        for obj in labels:
            obj_type = obj.get("type", "car").lower()
            if obj_type in ("trafficcone", "barrowlist"):
                continue
            loc = obj["3d_location"]
            dim = obj["3d_dimensions"]
            objects.append({
                "type": obj_type,
                "x": float(loc["x"]),
                "y": float(loc["y"]),
                "length": float(dim["l"]),
                "width": float(dim["w"]),
                "rotation": float(obj.get("rotation", obj.get("alpha", 0.0))),
                "occluded": obj.get("occluded_state", 0),
            })
        return objects

    def _compute_ego_velocity(self, idx: int) -> tuple[float, float, float]:
        """Compute ego velocity from consecutive frames. Returns (vx, vy, speed)."""
        if idx not in self.frame_to_prev:
            return 0.0, 0.0, 0.0

        prev_idx = self.frame_to_prev[idx]
        _, vid_curr = self.frames[idx]
        _, vid_prev = self.frames[prev_idx]

        pos_curr = self._load_ego_pose(vid_curr)
        pos_prev = self._load_ego_pose(vid_prev)
        if pos_curr is None or pos_prev is None:
            return 0.0, 0.0, 0.0

        ts_curr = self.veh_lookup[vid_curr]["timestamp_us"]
        ts_prev = self.veh_lookup[vid_prev]["timestamp_us"]
        dt = (ts_curr - ts_prev) / 1e6  # seconds
        if dt < 0.01:
            return 0.0, 0.0, 0.0

        vx = (pos_curr[0] - pos_prev[0]) / dt
        vy = (pos_curr[1] - pos_prev[1]) / dt
        speed = np.sqrt(vx ** 2 + vy ** 2)
        return float(vx), float(vy), float(speed)

    def _match_agent_velocity(self, curr_objs: list[dict], prev_objs: list[dict],
                              dt: float) -> dict[int, tuple[float, float]]:
        """Match agents between consecutive frames by nearest position, compute velocity."""
        velocities = {}  # curr_obj_index → (vx, vy)
        if not prev_objs or dt < 0.01:
            return velocities

        prev_positions = np.array([[o["x"], o["y"]] for o in prev_objs])
        for i, obj in enumerate(curr_objs):
            pos = np.array([obj["x"], obj["y"]])
            dists = np.linalg.norm(prev_positions - pos, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]

            # Only match if within reasonable distance (max 5m movement in ~0.1s → 50m/s)
            if min_dist < 5.0:
                prev = prev_objs[min_idx]
                vx = (obj["x"] - prev["x"]) / dt
                vy = (obj["y"] - prev["y"]) / dt
                velocities[i] = (float(vx), float(vy))

        return velocities

    def load_frame(self, idx: int) -> PerceptionResult:
        """Load one frame with REAL velocities."""
        label_path, vid = self.frames[idx]

        # Load ego pose and heading
        ego_pos = self._load_ego_pose(vid)
        ego_heading = self._load_ego_heading(vid)
        if ego_pos is None:
            ego_pos = np.array([0.0, 0.0, 0.0])

        # Compute ego velocity from consecutive frames
        ego_vx, ego_vy, ego_speed = self._compute_ego_velocity(idx)

        ego = VehicleState(
            id="ego",
            x=ego_pos[0], y=ego_pos[1],
            heading=ego_heading,
            velocity=ego_speed,
            vx=ego_vx, vy=ego_vy,
            length=4.5, width=1.8, mass=1500,
            vehicle_type="car",
        )

        # Load current and previous labels for agent velocity
        curr_objs = self._load_labels(label_path)
        agent_velocities = {}

        if idx in self.frame_to_prev:
            prev_idx = self.frame_to_prev[idx]
            prev_label_path, prev_vid = self.frames[prev_idx]
            prev_objs = self._load_labels(prev_label_path)
            ts_curr = self.veh_lookup[vid]["timestamp_us"]
            ts_prev = self.veh_lookup[prev_vid]["timestamp_us"]
            dt = (ts_curr - ts_prev) / 1e6
            agent_velocities = self._match_agent_velocity(curr_objs, prev_objs, dt)

        # Build agents
        agents = []
        for i, obj in enumerate(curr_objs):
            # Skip objects too close to ego (likely ego itself)
            dist = np.sqrt((obj["x"] - ego_pos[0]) ** 2 + (obj["y"] - ego_pos[1]) ** 2)
            if dist < 2.0:
                continue

            our_type, agent_type = DAIR_TYPE_MAP.get(obj["type"], ("car", AgentType.VEHICLE))
            heading = obj["rotation"]

            if i in agent_velocities:
                vx, vy = agent_velocities[i]
                speed = np.sqrt(vx ** 2 + vy ** 2)
            else:
                # No velocity data available — use 0 (honest: we don't know)
                vx, vy, speed = 0.0, 0.0, 0.0

            agent = Agent(
                state=VehicleState(
                    id=f"obj_{i:03d}",
                    x=obj["x"], y=obj["y"],
                    heading=heading,
                    velocity=speed,
                    vx=vx, vy=vy,
                    length=obj["length"],
                    width=obj["width"],
                    mass=DEFAULT_MASS.get(obj["type"], 1500),
                    vehicle_type=our_type,
                ),
                agent_type=agent_type,
                is_visible=obj["occluded"] == 0,
                confidence=0.9 if obj["occluded"] == 0 else 0.6,
                source="cooperative_v2i",
            )
            agents.append(agent)

        return PerceptionResult(
            timestamp=float(self.veh_lookup.get(vid, {}).get("timestamp_us", 0)) / 1e6,
            ego=ego,
            agents=agents,
        )

    def load_vehicle_side_only(self, idx: int) -> PerceptionResult:
        """Load frame with ONLY vehicle-side detections (for cooperative comparison).

        This gives the TRUE non-cooperative perception — only what the ego vehicle can see.
        """
        _, vid = self.frames[idx]

        # Load ego (same as cooperative)
        ego_pos = self._load_ego_pose(vid)
        ego_heading = self._load_ego_heading(vid)
        if ego_pos is None:
            ego_pos = np.array([0.0, 0.0, 0.0])
        ego_vx, ego_vy, ego_speed = self._compute_ego_velocity(idx)

        ego = VehicleState(
            id="ego", x=ego_pos[0], y=ego_pos[1],
            heading=ego_heading, velocity=ego_speed,
            vx=ego_vx, vy=ego_vy,
            length=4.5, width=1.8, mass=1500, vehicle_type="car",
        )

        # Load vehicle-side labels (NOT cooperative)
        veh_label_path = self.data_dir / "vehicle-side" / "label" / "lidar" / f"{vid}.json"
        if not veh_label_path.exists():
            return PerceptionResult(timestamp=0, ego=ego, agents=[])

        veh_labels = json.load(open(veh_label_path))

        # Vehicle-side labels are in vehicle LiDAR frame, need to transform to world
        # Load lidar_to_novatel and novatel_to_world
        cal_l2n_path = self.data_dir / "vehicle-side" / "calib" / "lidar_to_novatel" / f"{vid}.json"
        cal_n2w_path = self.data_dir / "vehicle-side" / "calib" / "novatel_to_world" / f"{vid}.json"

        if not cal_l2n_path.exists() or not cal_n2w_path.exists():
            return PerceptionResult(timestamp=0, ego=ego, agents=[])

        l2n = json.load(open(cal_l2n_path))
        n2w = json.load(open(cal_n2w_path))

        # Handle both formats: top-level rotation/translation or nested under transform
        if "transform" in l2n:
            R_l2n = np.array(l2n["transform"]["rotation"])
            t_l2n = np.array(l2n["transform"]["translation"]).flatten()
        else:
            R_l2n = np.array(l2n["rotation"])
            t_l2n = np.array(l2n["translation"]).flatten()

        R_n2w = np.array(n2w["rotation"])
        t_n2w = np.array(n2w["translation"]).flatten()

        agents = []
        for i, obj in enumerate(veh_labels):
            obj_type = obj.get("type", "car").lower()
            if obj_type in ("trafficcone", "barrowlist"):
                continue
            loc = obj["3d_location"]
            pos_lidar = np.array([float(loc["x"]), float(loc["y"]), float(loc["z"])])

            # Transform: lidar → novatel → world
            pos_novatel = R_l2n @ pos_lidar + t_l2n
            pos_world = R_n2w @ pos_novatel + t_n2w

            our_type, agent_type = DAIR_TYPE_MAP.get(obj_type, ("car", AgentType.VEHICLE))
            dim = obj["3d_dimensions"]

            agents.append(Agent(
                state=VehicleState(
                    id=f"veh_{i:03d}",
                    x=pos_world[0], y=pos_world[1],
                    heading=float(obj.get("rotation", 0.0)),
                    velocity=0.0, vx=0.0, vy=0.0,  # No velocity from single-side
                    length=float(dim["l"]), width=float(dim["w"]),
                    mass=DEFAULT_MASS.get(obj_type, 1500),
                    vehicle_type=our_type,
                ),
                agent_type=agent_type,
                is_visible=True,
                confidence=0.9,
                source="vehicle_side_only",
            ))

        return PerceptionResult(
            timestamp=float(self.veh_lookup.get(vid, {}).get("timestamp_us", 0)) / 1e6,
            ego=ego, agents=agents,
        )


if __name__ == "__main__":
    loader = DAIRv2xLoaderV2(
        "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"
    )
    # Verify velocity computation
    for idx in [0, 1, 100, 500, 1000]:
        if idx >= len(loader):
            break
        p = loader.load_frame(idx)
        has_prev = idx in loader.frame_to_prev
        n_with_vel = sum(1 for a in p.agents if a.state.velocity > 0.1)
        print(f"Frame {idx}: ego=({p.ego.x:.0f},{p.ego.y:.0f}) "
              f"speed={p.ego.velocity:.1f}m/s "
              f"agents={len(p.agents)} (vel>0: {n_with_vel}) "
              f"has_prev={has_prev}")

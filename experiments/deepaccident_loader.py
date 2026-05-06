"""DeepAccident dataset loader for safety constraint training and evaluation.

Uses ground-truth perception (3D labels) with:
- Visibility flag (visible=True/False) for blind spot modeling
- Collision countdown for supervision
- Other vehicle perspective for cooperative perception

Data format per line in label file:
  First line: ego_speed ego_yaw_rate
  Subsequent: type x y z length width height yaw vx vy id collision_countdown visible
"""
from __future__ import annotations

import os
import math
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
)

DATA_ROOT = Path("/raid/xuyifan/jiqiuyu/data/DeepAccident")

TYPE_MAP = {
    'car': ('car', AgentType.VEHICLE),
    'van': ('car', AgentType.VEHICLE),
    'truck': ('truck', AgentType.VEHICLE),
    'bus': ('bus', AgentType.VEHICLE),
    'motorcycle': ('motorcycle', AgentType.VEHICLE),
}


@dataclass
class DeepAccidentFrame:
    """One frame from DeepAccident with GT perception + collision labels."""
    perception: PerceptionResult  # GT perception (ego-centric)
    collision_countdown: dict  # agent_id → countdown (>0 = frames until collision)
    is_collision_frame: bool  # any agent has countdown > threshold
    max_collision_indicator: int  # max countdown across all agents
    ego_speed: float
    ego_yaw_rate: float
    scenario_name: str
    frame_idx: int
    is_accident_scenario: bool
    # Cooperative perception (other vehicle's view, if available)
    coop_perception: Optional[PerceptionResult] = None


def parse_label(label_path: str, include_invisible: bool = True) -> dict:
    """Parse a DeepAccident label file.

    Args:
        include_invisible: if True, include agents with visible=False (blind spot)
    """
    lines = open(label_path).readlines()
    if not lines:
        return {'ego_speed': 0, 'ego_yaw_rate': 0, 'agents': [], 'max_coll': 0}

    parts = lines[0].strip().split()
    ego_speed = float(parts[0]) if parts else 0
    ego_yaw_rate = float(parts[1]) if len(parts) > 1 else 0

    agents = []
    max_coll = 0
    for line in lines[1:]:
        p = line.strip().split()
        if len(p) < 13:
            continue
        coll = int(p[11])
        max_coll = max(max_coll, coll)
        visible = p[12] == 'True'
        if not include_invisible and not visible:
            continue
        agents.append({
            'type': p[0], 'x': float(p[1]), 'y': float(p[2]), 'z': float(p[3]),
            'length': float(p[4]), 'width': float(p[5]), 'height': float(p[6]),
            'yaw': float(p[7]), 'vx': float(p[8]), 'vy': float(p[9]),
            'id': int(p[10]), 'collision': coll, 'visible': visible,
        })

    return {'ego_speed': ego_speed, 'ego_yaw_rate': ego_yaw_rate,
            'agents': agents, 'max_coll': max_coll}


def label_to_perception(label: dict) -> PerceptionResult:
    """Convert parsed label to PerceptionResult."""
    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=label['ego_speed'],
        vx=label['ego_speed'], vy=0.0,
        yaw_rate=label['ego_yaw_rate'],
        length=4.5, width=1.8,
    )

    agents = []
    for a in label['agents']:
        if a['id'] == -100:  # ego marker
            continue
        our_type, agent_type = TYPE_MAP.get(a['type'], ('car', AgentType.VEHICLE))
        speed = math.sqrt(a['vx']**2 + a['vy']**2)
        agents.append(Agent(
            state=VehicleState(
                id=f"obj_{a['id']}",
                x=a['x'], y=a['y'], heading=a['yaw'],
                velocity=speed, vx=a['vx'], vy=a['vy'],
                length=a['length'], width=a['width'],
                vehicle_type=our_type,
            ),
            agent_type=agent_type,
            is_visible=a['visible'],
            confidence=1.0 if a['visible'] else 0.3,
        ))

    return PerceptionResult(timestamp=0, ego=ego, agents=agents)


class DeepAccidentLoader:
    """Load DeepAccident scenarios for training and evaluation."""

    def __init__(self, data_root: str = None, split: str = 'all',
                 include_invisible: bool = True, include_coop: bool = True):
        """
        Args:
            split: 'accident', 'normal', 'all'
            include_invisible: include invisible agents (blind spot GT)
            include_coop: load other_vehicle labels for cooperative perception
        """
        self.root = Path(data_root or DATA_ROOT)
        self.include_invisible = include_invisible
        self.include_coop = include_coop

        # Collect all scenarios
        self.scenarios = []
        for subtype_dir in sorted(self.root.iterdir()):
            if not subtype_dir.is_dir() or not subtype_dir.name.startswith('type'):
                continue
            is_accident = 'accident' in subtype_dir.name
            if split == 'accident' and not is_accident:
                continue
            if split == 'normal' and is_accident:
                continue

            ego_label_dir = subtype_dir / 'ego_vehicle' / 'label'
            coop_label_dir = subtype_dir / 'other_vehicle' / 'label'
            if not ego_label_dir.exists():
                continue

            for scenario_dir in sorted(ego_label_dir.iterdir()):
                if not scenario_dir.is_dir():
                    continue
                frames = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.txt')])
                coop_dir = coop_label_dir / scenario_dir.name if coop_label_dir.exists() else None

                # Parse meta for collision frame info
                meta_dir = subtype_dir / 'meta'
                meta_file = meta_dir / f"{scenario_dir.name}.txt"
                collision_frame = -1
                if meta_file.exists():
                    meta_lines = open(meta_file).readlines()
                    if meta_lines:
                        meta_parts = meta_lines[0].strip().split()
                        # Last field is collision frame (accident) or total frames (normal)
                        last_val = int(meta_parts[-1]) if meta_parts else -1
                        if is_accident and last_val > 0:
                            collision_frame = last_val

                self.scenarios.append({
                    'name': f"{subtype_dir.name}/{scenario_dir.name}",
                    'ego_dir': scenario_dir,
                    'coop_dir': coop_dir,
                    'frames': frames,
                    'is_accident': is_accident,
                    'collision_frame': collision_frame,
                })

        self.total_frames = sum(len(s['frames']) for s in self.scenarios)
        print(f"[DeepAccidentLoader] {len(self.scenarios)} scenarios, "
              f"{self.total_frames} frames, split={split}")

    def load_frame(self, scenario_idx: int, frame_idx: int) -> DeepAccidentFrame:
        """Load a single frame."""
        s = self.scenarios[scenario_idx]
        fname = s['frames'][frame_idx]

        # Ego perception (GT)
        ego_label = parse_label(str(s['ego_dir'] / fname), self.include_invisible)
        perception = label_to_perception(ego_label)

        # Collision info from meta (collision_frame = frame index where collision happens)
        collision_frame_idx = s.get('collision_frame', -1)
        frames_to_collision = collision_frame_idx - frame_idx if collision_frame_idx > 0 else -1
        is_near_collision = s['is_accident'] and 0 < frames_to_collision <= 20

        # Cooperative perception (other vehicle)
        coop_perception = None
        if self.include_coop and s['coop_dir'] and (s['coop_dir'] / fname).exists():
            coop_label = parse_label(str(s['coop_dir'] / fname), True)
            coop_perception = label_to_perception(coop_label)

        return DeepAccidentFrame(
            perception=perception,
            collision_countdown={},
            is_collision_frame=is_near_collision,
            max_collision_indicator=max(frames_to_collision, 0),
            ego_speed=ego_label['ego_speed'],
            ego_yaw_rate=ego_label['ego_yaw_rate'],
            scenario_name=s['name'],
            frame_idx=frame_idx,
            is_accident_scenario=s['is_accident'],
            coop_perception=coop_perception,
        )

    def iter_all_frames(self):
        """Iterate over all frames in all scenarios."""
        for si, s in enumerate(self.scenarios):
            for fi in range(len(s['frames'])):
                yield self.load_frame(si, fi)

    def get_training_data(self):
        """Get training data with collision labels from meta.

        Returns list of (perception, is_dangerous, frames_to_collision)
        """
        data = []
        for si, s in enumerate(self.scenarios):
            collision_frame = s.get('collision_frame', -1)
            for fi in range(len(s['frames'])):
                frame = self.load_frame(si, fi)
                ftc = collision_frame - fi if collision_frame > 0 else -1
                is_dangerous = s['is_accident'] and 0 < ftc <= 30  # Within 30 frames (~1.5s)
                ttc = ftc * 0.05 if ftc > 0 else 20.0
                data.append((frame.perception, is_dangerous, ttc))
        return data


if __name__ == "__main__":
    loader = DeepAccidentLoader(split='all')

    # Statistics
    n_accident_frames = 0
    n_normal_frames = 0
    n_collision_frames = 0
    for frame in loader.iter_all_frames():
        if frame.is_accident_scenario:
            n_accident_frames += 1
        else:
            n_normal_frames += 1
        if frame.is_collision_frame:
            n_collision_frames += 1

    print(f"\nStatistics:")
    print(f"  Accident frames: {n_accident_frames}")
    print(f"  Normal frames: {n_normal_frames}")
    print(f"  Frames with collision indicator: {n_collision_frames}")
    print(f"  Collision ratio: {n_collision_frames / (n_accident_frames + n_normal_frames):.1%}")

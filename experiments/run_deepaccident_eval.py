"""Evaluate safety constraint on DeepAccident dataset.

DeepAccident contains accident and normal driving scenarios.
We evaluate:
1. Can our safety constraint correctly identify high-risk frames in accident scenarios?
2. Does the constraint reduce collision metrics compared to baselines?
3. How does CoDriving's planned trajectory perform with/without safety constraint?

Metrics:
- Collision Prediction Accuracy: does constraint trigger CONSERVATIVE/MINIMUM_HARM before collision?
- Early Warning Time: how many frames before collision does constraint first trigger?
- False Alarm Rate: in normal scenarios, how often does constraint trigger unnecessarily?
- ADE/FDE with safety-modified trajectories (when applicable)
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, '/raid/xuyifan/jiqiuyu')

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, ConstraintMode,
)
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField


DATA_ROOT = Path("/raid/xuyifan/jiqiuyu/data/DeepAccident")


@dataclass
class ScenarioResult:
    scenario_name: str
    is_accident: bool
    n_frames: int
    # Safety constraint metrics
    first_warning_frame: int = -1  # Frame when CONSERVATIVE first triggered
    first_minharm_frame: int = -1  # Frame when MINIMUM_HARM first triggered
    collision_frame: int = -1      # Frame with highest collision indicator
    # Collision prediction
    warning_before_collision: bool = False
    early_warning_frames: int = 0
    # False alarm
    false_alarm_frames: int = 0
    # Constraint statistics
    total_constrained_frames: int = 0
    avg_constraint_ratio: float = 0.0


def parse_label_file(label_path: str) -> dict:
    """Parse DeepAccident label file.

    Returns dict with ego_speed, agents list, and collision info.
    """
    lines = open(label_path).readlines()
    if not lines:
        return {}

    # First line: ego speed and yaw rate
    first = lines[0].strip().split()
    ego_speed = float(first[0]) if first else 0.0
    ego_yaw_rate = float(first[1]) if len(first) > 1 else 0.0

    agents = []
    max_collision_indicator = 0

    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) < 13:
            continue

        obj_type = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        length, width, height = float(parts[4]), float(parts[5]), float(parts[6])
        yaw = float(parts[7])
        vx, vy = float(parts[8]), float(parts[9])
        obj_id = int(parts[10])
        collision_indicator = int(parts[11])  # >0 means involved in collision
        visible = parts[12] == 'True'

        max_collision_indicator = max(max_collision_indicator, collision_indicator)

        agents.append({
            'type': obj_type, 'x': x, 'y': y, 'z': z,
            'length': length, 'width': width, 'height': height,
            'yaw': yaw, 'vx': vx, 'vy': vy,
            'id': obj_id, 'collision': collision_indicator,
            'visible': visible,
        })

    return {
        'ego_speed': ego_speed,
        'ego_yaw_rate': ego_yaw_rate,
        'agents': agents,
        'max_collision_indicator': max_collision_indicator,
    }


def label_to_perception(label: dict) -> PerceptionResult:
    """Convert DeepAccident label to PerceptionResult."""
    ego_speed = label.get('ego_speed', 5.0)

    ego = VehicleState(
        id="ego", x=0.0, y=0.0, heading=0.0,
        velocity=ego_speed, vx=ego_speed, vy=0.0,
        length=4.5, width=1.8,
    )

    agents = []
    type_map = {
        'car': ('car', AgentType.VEHICLE),
        'van': ('car', AgentType.VEHICLE),
        'motorcycle': ('motorcycle', AgentType.VEHICLE),
        'truck': ('truck', AgentType.VEHICLE),
        'bus': ('bus', AgentType.VEHICLE),
    }

    for a in label.get('agents', []):
        if a['id'] == -100:  # Ego vehicle marker
            continue
        our_type, agent_type = type_map.get(a['type'], ('car', AgentType.VEHICLE))
        speed = math.sqrt(a['vx']**2 + a['vy']**2)

        agents.append(Agent(
            state=VehicleState(
                id=f"obj_{a['id']}",
                x=a['x'], y=a['y'],
                heading=a['yaw'],
                velocity=speed, vx=a['vx'], vy=a['vy'],
                length=a['length'], width=a['width'],
                mass=1500 if our_type == 'car' else 8000,
                vehicle_type=our_type,
            ),
            agent_type=agent_type,
            is_visible=a['visible'],
        ))

    return PerceptionResult(timestamp=0, ego=ego, agents=agents)


def evaluate_scenario(scenario_dir: Path, is_accident: bool, safety_module) -> ScenarioResult:
    """Evaluate one scenario."""
    label_dir = scenario_dir
    frames = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])

    result = ScenarioResult(
        scenario_name=scenario_dir.parent.parent.name + '/' + scenario_dir.name,
        is_accident=is_accident,
        n_frames=len(frames),
    )

    constraint_ratios = []

    # Find collision frame (frame with highest collision indicator)
    max_coll = 0
    for i, fname in enumerate(frames):
        label = parse_label_file(str(label_dir / fname))
        coll = label.get('max_collision_indicator', 0)
        if coll > max_coll:
            max_coll = coll
            result.collision_frame = i

    # Evaluate safety constraint frame by frame
    for i, fname in enumerate(frames):
        label = parse_label_file(str(label_dir / fname))
        if not label.get('agents'):
            continue

        perception = label_to_perception(label)

        try:
            safe = safety_module.constrain(perception)
        except:
            continue

        mode = safe.mode

        if mode in (ConstraintMode.CONSERVATIVE, ConstraintMode.MINIMUM_HARM):
            result.total_constrained_frames += 1

            if result.first_warning_frame < 0:
                result.first_warning_frame = i

            if mode == ConstraintMode.MINIMUM_HARM and result.first_minharm_frame < 0:
                result.first_minharm_frame = i

            if not is_accident:
                result.false_alarm_frames += 1

        # Compute constraint ratio
        from shapely.geometry import Polygon
        if len(safe.feasible_region) >= 3:
            area = Polygon(safe.feasible_region).area
            constraint_ratios.append(1.0 - area / 1500.0)  # Approximate

    # Check if warning came before collision
    if is_accident and result.collision_frame > 0 and result.first_warning_frame >= 0:
        if result.first_warning_frame < result.collision_frame:
            result.warning_before_collision = True
            result.early_warning_frames = result.collision_frame - result.first_warning_frame

    result.avg_constraint_ratio = np.mean(constraint_ratios) if constraint_ratios else 0.0

    return result


def main():
    print("=" * 70)
    print("DeepAccident Safety Constraint Evaluation")
    print("=" * 70)

    # Setup methods
    methods = {
        "Ours-Rule": SafetyConstraintModule(),
        "RSS": RSSOnly(),
        "APF [Rasekhipour17]": RiskPotentialField(),
    }

    # Collect scenarios
    scenarios = []
    for subtype_dir in sorted(DATA_ROOT.iterdir()):
        if not subtype_dir.is_dir() or not subtype_dir.name.startswith('type'):
            continue
        is_accident = 'accident' in subtype_dir.name
        label_base = subtype_dir / 'ego_vehicle' / 'label'
        if not label_base.exists():
            continue
        for scenario in sorted(label_base.iterdir()):
            if scenario.is_dir():
                scenarios.append((scenario, is_accident))

    print(f"Total scenarios: {len(scenarios)}")
    accident_count = sum(1 for _, a in scenarios if a)
    normal_count = len(scenarios) - accident_count
    print(f"  Accident: {accident_count}, Normal: {normal_count}")

    # Limit for speed (can increase later)
    max_scenarios = min(len(scenarios), 100)
    scenarios = scenarios[:max_scenarios]
    print(f"Evaluating {max_scenarios} scenarios...")

    for method_name, module in methods.items():
        print(f"\n--- {method_name} ---")

        results = []
        t0 = time.time()

        for i, (scenario_dir, is_accident) in enumerate(scenarios):
            if i % 20 == 0:
                print(f"  [{i}/{max_scenarios}] {time.time()-t0:.0f}s")

            r = evaluate_scenario(scenario_dir, is_accident, module)
            results.append(r)

        # Summarize
        acc_results = [r for r in results if r.is_accident]
        norm_results = [r for r in results if not r.is_accident]

        # Collision prediction metrics (accident scenarios)
        if acc_results:
            detection_rate = sum(1 for r in acc_results if r.first_warning_frame >= 0) / len(acc_results)
            early_warning = [r.early_warning_frames for r in acc_results if r.warning_before_collision]
            avg_early = np.mean(early_warning) if early_warning else 0
            minharm_rate = sum(1 for r in acc_results if r.first_minharm_frame >= 0) / len(acc_results)
        else:
            detection_rate, avg_early, minharm_rate = 0, 0, 0

        # False alarm metrics (normal scenarios)
        if norm_results:
            false_alarm_rate = sum(1 for r in norm_results if r.false_alarm_frames > 0) / len(norm_results)
            avg_false_frames = np.mean([r.false_alarm_frames for r in norm_results])
        else:
            false_alarm_rate, avg_false_frames = 0, 0

        avg_constraint = np.mean([r.avg_constraint_ratio for r in results])

        print(f"\n  {method_name} Results:")
        print(f"    Accident Detection Rate:  {detection_rate:.1%} ({sum(1 for r in acc_results if r.first_warning_frame>=0)}/{len(acc_results)})")
        print(f"    Avg Early Warning Frames: {avg_early:.1f}")
        print(f"    Min-Harm Trigger Rate:    {minharm_rate:.1%}")
        print(f"    False Alarm Rate:         {false_alarm_rate:.1%}")
        print(f"    Avg False Alarm Frames:   {avg_false_frames:.1f}")
        print(f"    Avg Constraint Ratio:     {avg_constraint:.1%}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

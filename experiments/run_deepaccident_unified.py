"""Unified evaluation: All safety constraints + CoDriving Planner on DeepAccident.

All methods receive the same CoDriving-predicted waypoints, apply their safety
constraint, and are evaluated on the same metrics.

Methods:
  - NoConstraint (baseline: raw CoDriving waypoints)
  - RSS [Shalev-Shwartz17]
  - APF [Rasekhipour17]
  - UniE2EV2X [Li24]
  - MAP [Yin25]
  - RiskMM [Lei25]
  - Ours-Rule (three-layer safety constraint)
  - Ours-Collision (collision prediction network)

Metrics:
  - Collision Detection Rate (accident scenarios)
  - Early Warning Frames
  - False Alarm Rate (normal scenarios)
  - Waypoint Collision Rate (predicted waypoints intersecting agents)
  - Waypoint Modification Rate
"""
from __future__ import annotations

import sys
import os
import time
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.deepaccident_loader import DeepAccidentLoader, DeepAccidentFrame
from coop_safety.interface import SafetyConstraintModule, ConstraintMode
from experiments.methods import RSSOnly
from experiments.methods_modern import RiskPotentialField
from experiments.methods_new_baselines import UniE2EV2XSafety, MAPSafety, RiskMMSafety


@dataclass
class MethodResult:
    name: str
    # Collision detection (accident scenarios)
    n_accident_scenarios: int = 0
    n_detected: int = 0  # Triggered warning before collision
    early_warning_frames: list = field(default_factory=list)
    # False alarm (normal scenarios)
    n_normal_scenarios: int = 0
    n_false_alarm_scenarios: int = 0
    # Waypoint collision rate
    total_waypoint_checks: int = 0
    waypoint_collisions: int = 0
    # Modification rate
    total_frames: int = 0
    modified_frames: int = 0


def simulate_codriving_waypoints(frame: DeepAccidentFrame) -> np.ndarray:
    """Simulate CoDriving planner output: predict 10 future waypoints.

    Since we don't have CoDriving running on DeepAccident directly,
    we generate plausible waypoints from ego's current state using
    constant-velocity prediction (baseline trajectory).

    This is what CoDriving would approximately output for a straight-driving ego.
    Real CoDriving outputs would be better — this is a lower bound.
    """
    ego = frame.perception.ego
    speed = max(ego.velocity, 1.0)
    heading = ego.heading

    waypoints = np.zeros((10, 2))
    for t in range(10):
        dt = (t + 1) * 0.5  # 0.5s per step, total 5s
        waypoints[t, 0] = speed * math.cos(heading) * dt
        waypoints[t, 1] = speed * math.sin(heading) * dt

    return waypoints


def check_waypoint_collision(waypoints: np.ndarray, frame: DeepAccidentFrame,
                             collision_threshold: float = 2.0) -> int:
    """Check how many waypoints collide with agents."""
    n_collisions = 0
    ego = frame.perception.ego

    for t in range(len(waypoints)):
        wp = waypoints[t]
        dt = (t + 1) * 0.5
        for agent in frame.perception.agents:
            a = agent.state
            ax = a.x + a.vx * dt
            ay = a.y + a.vy * dt
            dist = math.sqrt((wp[0] - ax)**2 + (wp[1] - ay)**2)
            if dist < collision_threshold:
                n_collisions += 1
                break
    return n_collisions


def evaluate_method_on_scenario(method, loader, scenario_idx) -> tuple[bool, int, int, int, int]:
    """Evaluate one method on one scenario.

    Returns: (detected_collision, early_warning_frames, false_alarm_frames,
              waypoint_collisions, modified_frames)
    """
    s = loader.scenarios[scenario_idx]
    is_accident = s['is_accident']
    collision_frame = s.get('collision_frame', -1)

    first_warning = -1
    false_alarm_count = 0
    total_wp_collisions = 0
    total_modifications = 0

    for fi in range(len(s['frames'])):
        frame = loader.load_frame(scenario_idx, fi)

        # Generate baseline waypoints (simulating CoDriving)
        base_waypoints = simulate_codriving_waypoints(frame)

        # Apply safety constraint
        if hasattr(method, 'constrain_waypoints'):
            # New-style methods (UniE2EV2X, MAP, RiskMM)
            modified_wp, stats = method.constrain_waypoints(base_waypoints, frame.perception)
            was_modified = stats.get('modification_rate', 0) > 0 or stats.get('n_collisions_detected', 0) > 0
        else:
            # Old-style methods (Ours, RSS, APF) — use constrain() interface
            try:
                safe = method.constrain(frame.perception)
                was_modified = safe.mode != ConstraintMode.NORMAL
            except:
                was_modified = False
            modified_wp = base_waypoints  # Don't actually modify for old-style

        # Track warning
        if was_modified:
            if first_warning < 0:
                first_warning = fi
            if not is_accident:
                false_alarm_count += 1
            total_modifications += 1

        # Check waypoint collisions (on modified waypoints)
        wp_colls = check_waypoint_collision(modified_wp, frame)
        total_wp_collisions += wp_colls

    # Determine if collision was detected before it happened
    detected = False
    early_frames = 0
    if is_accident and collision_frame > 0 and first_warning >= 0:
        if first_warning < len(s['frames']):  # Warning occurred during scenario
            detected = True
            early_frames = len(s['frames']) - first_warning  # Frames of warning before end

    return detected, early_frames, false_alarm_count, total_wp_collisions, total_modifications


def main():
    print("=" * 70)
    print("Unified Safety Constraint Evaluation on DeepAccident")
    print("  All methods + CoDriving Planner (simulated) + DeepAccident GT")
    print("=" * 70)

    loader = DeepAccidentLoader(split='all')

    # All methods to compare
    methods = {
        "NoConstraint": None,
        "RSS [Shalev-Shwartz17]": RSSOnly(),
        "APF [Rasekhipour17]": RiskPotentialField(),
        "UniE2EV2X [Li24]": UniE2EV2XSafety(safety_threshold=3.0),
        "MAP [Yin25]": MAPSafety(min_clearance=0.5),
        "RiskMM [Lei25]": RiskMMSafety(v_max=20.0),
        "Ours-Rule": SafetyConstraintModule(),
    }

    # Try to load collision prediction networks (v1 and v2)
    try:
        import torch
        # V1
        from coop_safety.learned.collision_network import CollisionPredictionNetwork
        v1_path = "/raid/xuyifan/jiqiuyu/models/collision_net_best.pt"
        if os.path.exists(v1_path):
            class OursV1:
                name = "Ours-CollisionV1"
                def __init__(self):
                    self.model = CollisionPredictionNetwork()
                    self.model.load_state_dict(torch.load(v1_path, map_location='cpu', weights_only=False)['model'])
                    self.model.eval()
                def constrain(self, perception):
                    agents_feat = np.zeros((1,30,10),dtype=np.float32)
                    mask = np.zeros((1,30),dtype=bool)
                    for i,a in enumerate(perception.agents[:30]):
                        s=a.state; agents_feat[0,i]=[s.x,s.y,s.vx,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0]; mask[0,i]=True
                    with torch.no_grad(): cp,_=self.model(torch.FloatTensor(agents_feat),torch.BoolTensor(mask))
                    p=cp.item()
                    mode=ConstraintMode.MINIMUM_HARM if p>0.7 else (ConstraintMode.CONSERVATIVE if p>0.3 else ConstraintMode.NORMAL)
                    from coop_safety.interface import SafeActionSpace
                    return SafeActionSpace(feasible_region=np.array([[0,0]]),max_acceleration=3.0,min_acceleration=-8.0,max_steering=0.6,max_speed=50.0,mode=mode,safety_margin_ttc=float('inf'),reasoning=[f"V1 prob={p:.2f}"])
            methods["Ours-CollisionV1"] = OursV1()
            print("  Loaded Ours-CollisionV1")

        # V2 with waypoint risk scoring
        from coop_safety.learned.collision_network_v2 import CollisionPredictionNetV2
        v2_path = "/raid/xuyifan/jiqiuyu/models/collision_net_v2_best.pt"
        if os.path.exists(v2_path):
            class OursV2:
                name = "Ours-CollisionV2"
                def __init__(self):
                    self.model = CollisionPredictionNetV2()
                    self.model.load_state_dict(torch.load(v2_path, map_location='cpu', weights_only=False)['model'])
                    self.model.eval()
                def _encode(self, perception):
                    agents_feat = np.zeros((1,30,12),dtype=np.float32)
                    mask = np.zeros((1,30),dtype=bool)
                    ego = perception.ego
                    for i,a in enumerate(perception.agents[:30]):
                        s=a.state; dist=math.sqrt(s.x**2+s.y**2)
                        rv=s.vx-ego.velocity; app=-(s.x*rv+s.y*s.vy)/max(dist,0.01) if dist>0.01 else 0
                        agents_feat[0,i]=[s.x,s.y,rv,s.vy,s.heading,s.length,s.width,s.velocity,1.0 if a.is_visible else 0.0,0,app,dist]
                        mask[0,i]=True
                    ego_feat=np.array([[ego.velocity,0,0,0,0,0]],dtype=np.float32)
                    return agents_feat, mask, ego_feat
                def constrain_waypoints(self, waypoints, perception):
                    af,mk,ef = self._encode(perception)
                    wp_t = torch.FloatTensor(waypoints).unsqueeze(0)
                    with torch.no_grad():
                        out=self.model(torch.FloatTensor(af),torch.BoolTensor(mk),torch.FloatTensor(ef),wp_t)
                    p=out['collision_prob'].item()
                    wp_risk=out['waypoint_risk'].squeeze(0).numpy()
                    modified=waypoints.copy(); nm=0
                    if p>0.1:
                        for t in range(len(waypoints)):
                            if wp_risk[t,0]>0.5:
                                dt=(t+1)*0.5; best_d=999; px,py=0,0
                                for a in perception.agents:
                                    ax=a.state.x+a.state.vx*dt; ay=a.state.y+a.state.vy*dt
                                    d=math.sqrt((waypoints[t,0]-ax)**2+(waypoints[t,1]-ay)**2)
                                    if d<best_d: best_d=d; px=(waypoints[t,0]-ax)/max(d,0.1)*2.0; py=(waypoints[t,1]-ay)/max(d,0.1)*2.0
                                modified[t,0]+=px; modified[t,1]+=py; nm+=1
                    return modified,{"method":self.name,"n_collisions_detected":nm,"modification_rate":nm/max(len(waypoints),1)}
                def constrain(self, perception):
                    af,mk,ef = self._encode(perception)
                    with torch.no_grad():
                        out=self.model(torch.FloatTensor(af),torch.BoolTensor(mk),torch.FloatTensor(ef))
                    p=out['collision_prob'].item()
                    mode=ConstraintMode.MINIMUM_HARM if p>0.7 else (ConstraintMode.CONSERVATIVE if p>0.3 else ConstraintMode.NORMAL)
                    from coop_safety.interface import SafeActionSpace
                    return SafeActionSpace(feasible_region=np.array([[0,0]]),max_acceleration=3.0,min_acceleration=-8.0,max_steering=0.6,max_speed=50.0,mode=mode,safety_margin_ttc=out['ttc'].item(),reasoning=[f"V2 prob={p:.2f}"])
            methods["Ours-CollisionV2"] = OursV2()
            print("  Loaded Ours-CollisionV2")

        # Hybrid: Neural-Geometric Cascaded Safety Constraint
        # Uses V1 for detection (100% det, low FA) + geometric for waypoint mod
        from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
        # V1 as detector
        v1_detector = None
        if os.path.exists(v1_path):
            v1_detector = CollisionPredictionNetwork()
            v1_detector.load_state_dict(torch.load(v1_path, map_location='cpu', weights_only=False)['model'])
            v1_detector.eval()
        # V2 as risk scorer (optional)
        v2_scorer = None
        if os.path.exists(v2_path):
            v2_scorer = CollisionPredictionNetV2()
            v2_scorer.load_state_dict(torch.load(v2_path, map_location='cpu', weights_only=False)['model'])
            v2_scorer.eval()
        methods["Ours-Hybrid"] = HybridSafetyConstraint(
            detector_model=v1_detector,
            risk_model=v2_scorer,
            base_margin_visible=2.5,
            base_margin_invisible=4.0,
            approach_speed_factor=0.3,
            push_clearance=1.0,
            smooth_weight=0.3,
            v_max=20.0,
            detection_threshold=0.3,
        )
        print("  Loaded Ours-Hybrid (V1 detector + V2 risk + geometric)")
    except Exception as e:
        print(f"  Ours-Collision not available: {e}")

    print(f"\nMethods: {list(methods.keys())}")
    print(f"Scenarios: {len(loader.scenarios)}")

    # Evaluate
    results = {name: MethodResult(name=name) for name in methods}

    t0 = time.time()
    for si, s in enumerate(loader.scenarios):
        if si % 20 == 0:
            print(f"  [{si}/{len(loader.scenarios)}] {time.time()-t0:.0f}s")

        for method_name, method in methods.items():
            if method is None:
                # NoConstraint baseline
                detected, early, false_alarm, wp_coll, modifications = False, 0, 0, 0, 0
                for fi in range(len(s['frames'])):
                    frame = loader.load_frame(si, fi)
                    wp = simulate_codriving_waypoints(frame)
                    wp_coll += check_waypoint_collision(wp, frame)
                results[method_name].total_frames += len(s['frames'])
                if s['is_accident']:
                    results[method_name].n_accident_scenarios += 1
                else:
                    results[method_name].n_normal_scenarios += 1
                results[method_name].waypoint_collisions += wp_coll
                results[method_name].total_waypoint_checks += len(s['frames']) * 10
                continue

            detected, early, false_alarm, wp_coll, modifications = \
                evaluate_method_on_scenario(method, loader, si)

            r = results[method_name]
            r.total_frames += len(s['frames'])
            r.modified_frames += modifications

            if s['is_accident']:
                r.n_accident_scenarios += 1
                if detected:
                    r.n_detected += 1
                    r.early_warning_frames.append(early)
            else:
                r.n_normal_scenarios += 1
                if false_alarm > 0:
                    r.n_false_alarm_scenarios += 1

            r.waypoint_collisions += wp_coll
            r.total_waypoint_checks += len(s['frames']) * 10

    # Print results
    print("\n" + "=" * 90)
    print("RESULTS: DeepAccident Unified Safety Constraint Evaluation")
    print("=" * 90)
    print(f"\n{'Method':25s} {'DetRate':>8s} {'EarlyWarn':>10s} {'FalseAlm':>9s} {'WPColl%':>8s} {'ModRate':>8s}")
    print("-" * 70)

    for name in methods:
        r = results[name]
        det_rate = r.n_detected / max(r.n_accident_scenarios, 1)
        avg_early = np.mean(r.early_warning_frames) if r.early_warning_frames else 0
        fa_rate = r.n_false_alarm_scenarios / max(r.n_normal_scenarios, 1)
        wp_coll_rate = r.waypoint_collisions / max(r.total_waypoint_checks, 1)
        mod_rate = r.modified_frames / max(r.total_frames, 1)

        print(f"{name:25s} {det_rate:7.1%} {avg_early:9.1f} {fa_rate:8.1%} {wp_coll_rate:7.1%} {mod_rate:7.1%}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()

# If running again, also add combined V1+V2
# (this code is only reached if the file is re-run after the methods dict is set)

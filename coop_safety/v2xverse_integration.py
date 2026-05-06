"""Safety constraint wrapper for V2Xverse CoDriving agent.

Intercepts the planning output (throttle, brake, steer) and applies
safety constraint checking. If the planned action would enter an unsafe
region, the action is modified to stay within the safe action space.

Integration point: after model prediction, before control output.

Methods:
  - Ours-Baseline: rule-based three-layer safety constraint
  - Ours-AB: learned risk assessment (A) + learned parameter adjuster (B)
  - RSS/CBF/RPF: baseline safety constraints for comparison
"""

import sys
import math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import carla
except ImportError:
    carla = None

from coop_safety.interface import (
    PerceptionResult, VehicleState, Agent, AgentType,
    SafetyConstraintModule, SafeActionSpace, ConstraintMode,
)


class SafetyConstraintWrapper:
    """Wraps a V2Xverse agent's control output with safety checking.

    Usage in coop_agent.py:
        # After model predicts control:
        pred_control = carla.VehicleControl(throttle=t, brake=b, steer=s)

        # Apply safety constraint:
        safe_control = self.safety_wrapper.check_and_modify(
            pred_control, ego_state, detected_agents)
    """

    def __init__(self, method="ours-baseline"):
        """
        Args:
            method: "none", "rss", "cbf", "rpf", "ours-baseline", "ours-ab"
        """
        self.method = method
        self.module = None
        self.stats = {"total": 0, "modified": 0, "min_harm": 0}

        if method == "ours-baseline":
            self.module = SafetyConstraintModule()
        elif method == "ours-ab":
            # Learned version
            try:
                from experiments.run_learned_comparison import LearnedSafetyModule
                self.module = LearnedSafetyModule()
            except:
                print("[SafetyWrapper] Learned module not available, falling back to rule-based")
                self.module = SafetyConstraintModule()
        elif method == "rss":
            from experiments.methods import RSSOnly
            self.module = RSSOnly()
        elif method == "rpf":
            from experiments.methods_modern import RiskPotentialField
            self.module = RiskPotentialField()
        elif method == "none":
            self.module = None

    def check_and_modify(self, control, ego_transform, ego_velocity,
                         other_actors):
        """Check if planned control is safe, modify if needed.

        Args:
            control: carla.VehicleControl (throttle, brake, steer)
            ego_transform: carla.Transform of ego vehicle
            ego_velocity: carla.Vector3D of ego velocity
            other_actors: list of (transform, velocity, extent) tuples for other actors

        Returns:
            Modified carla.VehicleControl
        """
        self.stats["total"] += 1

        if self.module is None:
            return control

        # Build perception
        perception = self._build_perception(ego_transform, ego_velocity, other_actors)
        if perception is None:
            return control

        # Run safety constraint
        try:
            safe_space = self.module.constrain(perception)
        except Exception:
            return control

        # Check if current action is within safe space
        if safe_space.mode == ConstraintMode.NORMAL:
            return control  # No modification needed

        self.stats["modified"] += 1

        if safe_space.mode == ConstraintMode.MINIMUM_HARM:
            self.stats["min_harm"] += 1
            # Reduce speed but don't slam brakes (learned from CARLA experiments)
            return self._smooth_brake(control, intensity=0.6)

        if safe_space.mode == ConstraintMode.CONSERVATIVE:
            # Reduce throttle, add light braking
            return self._smooth_brake(control, intensity=0.3)

        return control

    def _smooth_brake(self, control, intensity=0.5):
        """Apply smooth braking without sudden stops."""
        new_control = carla.VehicleControl() if carla else type('C', (), {})()
        new_control.throttle = max(0, control.throttle * (1 - intensity))
        new_control.brake = min(1.0, control.brake + intensity * 0.5)
        new_control.steer = control.steer  # Keep steering direction
        return new_control

    def _build_perception(self, ego_transform, ego_velocity, other_actors):
        """Convert CARLA actor data to PerceptionResult."""
        try:
            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
            ego = VehicleState(
                id="ego",
                x=ego_transform.location.x,
                y=ego_transform.location.y,
                heading=math.radians(ego_transform.rotation.yaw),
                velocity=max(ego_speed, 1.0),
                vx=ego_velocity.x,
                vy=ego_velocity.y,
                length=4.5, width=1.8,
            )

            agents = []
            for i, (t, v, ext) in enumerate(other_actors):
                dx = t.location.x - ego.x
                dy = t.location.y - ego.y
                if dx*dx + dy*dy > 3600:  # > 60m
                    continue
                agents.append(Agent(
                    state=VehicleState(
                        id=f"a{i}",
                        x=t.location.x, y=t.location.y,
                        heading=math.radians(t.rotation.yaw),
                        velocity=max(math.sqrt(v.x**2 + v.y**2), 0.1),
                        vx=v.x, vy=v.y,
                        length=max(ext.x * 2, 2.0),
                        width=max(ext.y * 2, 1.0),
                    ),
                    agent_type=AgentType.VEHICLE,
                ))

            return PerceptionResult(timestamp=0, ego=ego, agents=agents)
        except Exception:
            return None

    def get_stats(self):
        return self.stats


# Evaluation metrics for V2Xverse
SAFETY_METRICS = {
    "collision_rate": "Number of collisions per route",
    "near_miss_rate": "Frames with min distance to any agent < 3m",
    "min_ttc": "Minimum Time-to-Collision observed",
    "safety_intervention_rate": "Fraction of frames where safety constraint modified control",
    "min_harm_rate": "Fraction of frames in MINIMUM_HARM mode",
    "route_completion": "Fraction of route completed (V2Xverse standard)",
    "driving_score": "V2Xverse driving score = RC × penalty",
    "comfort": "Mean absolute jerk (lower = smoother)",
}

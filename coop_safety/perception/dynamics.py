"""Vehicle dynamics models for trajectory prediction and feasible region computation.

Implements bicycle model and kinematic constraints for different vehicle types.
Corresponds to research plan step 2.1.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DynamicsParams:
    """Vehicle dynamics parameters derived from vehicle properties."""
    max_acceleration: float      # m/s^2
    max_deceleration: float      # m/s^2 (positive value)
    max_steering_angle: float    # radians
    max_steering_rate: float     # rad/s
    wheelbase: float             # meters (front to rear axle)
    max_speed: float             # m/s
    min_turn_radius: float       # meters


def get_dynamics_params(vehicle_type: str, length: float, mass: float) -> DynamicsParams:
    """Match vehicle dynamics parameters based on type and size (Plan 2.1).

    Args:
        vehicle_type: "car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"
        length: Vehicle length in meters
        mass: Vehicle mass in kg

    Returns:
        DynamicsParams for the vehicle
    """
    # Wheelbase approximation: ~60% of vehicle length for cars
    wheelbase_ratio = {"car": 0.6, "truck": 0.55, "bus": 0.55,
                       "motorcycle": 0.65, "bicycle": 0.65, "pedestrian": 0.3}
    ratio = wheelbase_ratio.get(vehicle_type, 0.6)
    wheelbase = max(length * ratio, 0.5)

    params_table = {
        "car": DynamicsParams(
            max_acceleration=3.0, max_deceleration=8.0,
            max_steering_angle=np.radians(35), max_steering_rate=np.radians(60),
            wheelbase=wheelbase, max_speed=50.0,
            min_turn_radius=wheelbase / np.tan(np.radians(35)),
        ),
        "truck": DynamicsParams(
            max_acceleration=1.5, max_deceleration=5.0,
            max_steering_angle=np.radians(25), max_steering_rate=np.radians(30),
            wheelbase=wheelbase, max_speed=30.0,
            min_turn_radius=wheelbase / np.tan(np.radians(25)),
        ),
        "bus": DynamicsParams(
            max_acceleration=1.2, max_deceleration=4.5,
            max_steering_angle=np.radians(25), max_steering_rate=np.radians(25),
            wheelbase=wheelbase, max_speed=25.0,
            min_turn_radius=wheelbase / np.tan(np.radians(25)),
        ),
        "motorcycle": DynamicsParams(
            max_acceleration=5.0, max_deceleration=9.0,
            max_steering_angle=np.radians(40), max_steering_rate=np.radians(90),
            wheelbase=wheelbase, max_speed=60.0,
            min_turn_radius=wheelbase / np.tan(np.radians(40)),
        ),
        "bicycle": DynamicsParams(
            max_acceleration=2.0, max_deceleration=4.0,
            max_steering_angle=np.radians(45), max_steering_rate=np.radians(120),
            wheelbase=wheelbase, max_speed=10.0,
            min_turn_radius=wheelbase / np.tan(np.radians(45)),
        ),
        "pedestrian": DynamicsParams(
            max_acceleration=2.5, max_deceleration=3.0,
            max_steering_angle=np.pi, max_steering_rate=np.pi * 2,
            wheelbase=0.3, max_speed=2.0, min_turn_radius=0.1,
        ),
    }
    return params_table.get(vehicle_type, params_table["car"])


class BicycleModel:
    """Kinematic bicycle model for vehicle motion prediction.

    State: [x, y, heading, velocity]
    Control: [acceleration, steering_angle]

    Used for:
    - Predicting future states given control inputs
    - Computing reachable sets (feasible regions)
    """

    def __init__(self, params: DynamicsParams):
        self.params = params

    def step(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """Propagate state forward by dt seconds.

        Args:
            state: [x, y, heading, velocity]
            control: [acceleration, steering_angle]
            dt: Time step in seconds

        Returns:
            New state [x, y, heading, velocity]
        """
        x, y, theta, v = state
        a, delta = control

        # Clamp controls
        a = np.clip(a, -self.params.max_deceleration, self.params.max_acceleration)
        delta = np.clip(delta, -self.params.max_steering_angle, self.params.max_steering_angle)

        # Bicycle model kinematics
        L = self.params.wheelbase
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / L) * np.tan(delta) * dt
        v_new = np.clip(v + a * dt, 0, self.params.max_speed)

        return np.array([x_new, y_new, theta_new, v_new])

    def predict_trajectory(self, state: np.ndarray, control: np.ndarray,
                           horizon: float, dt: float = 0.1) -> np.ndarray:
        """Predict trajectory under constant control.

        Args:
            state: Initial [x, y, heading, velocity]
            control: Constant [acceleration, steering_angle]
            horizon: Prediction horizon in seconds
            dt: Time step

        Returns:
            (T, 4) array of states over time
        """
        steps = int(horizon / dt)
        trajectory = np.zeros((steps + 1, 4))
        trajectory[0] = state
        for i in range(steps):
            trajectory[i + 1] = self.step(trajectory[i], control, dt)
        return trajectory

    def compute_reachable_set(self, state: np.ndarray, horizon: float,
                              dt: float = 0.1, n_samples: int = 12) -> np.ndarray:
        """Compute reachable positions by sampling control inputs.

        Samples different acceleration and steering combinations to find
        the envelope of reachable positions at time=horizon.

        Args:
            state: Current [x, y, heading, velocity]
            horizon: Time horizon in seconds
            dt: Simulation time step
            n_samples: Number of samples per control dimension

        Returns:
            (N, 2) array of reachable boundary positions
        """
        accel_range = np.linspace(-self.params.max_deceleration,
                                  self.params.max_acceleration, n_samples)
        steer_range = np.linspace(-self.params.max_steering_angle,
                                  self.params.max_steering_angle, n_samples)

        endpoints = []
        for a in accel_range:
            for delta in steer_range:
                traj = self.predict_trajectory(state, np.array([a, delta]), horizon, dt)
                endpoints.append(traj[-1, :2])

        points = np.array(endpoints)

        # Compute convex hull for the reachable boundary
        if len(points) < 3:
            return points

        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return points[hull.vertices]
        except Exception:
            return points

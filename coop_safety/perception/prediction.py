from __future__ import annotations
"""Trajectory prediction for traffic participants.

Provides multiple prediction strategies from simple (constant velocity)
to moderate (constant turn rate and acceleration, CTRA).
Corresponds to research plan step 2.3 (global prediction).
"""

import numpy as np
from typing import Optional

from ..interface import Agent, VehicleState


def predict_constant_velocity(state: VehicleState, horizon: float,
                              dt: float = 0.1) -> np.ndarray:
    """Predict trajectory assuming constant velocity and heading.

    Simplest predictor. Used as fallback and for low-confidence detections.

    Args:
        state: Current vehicle state
        horizon: Prediction horizon (seconds)
        dt: Time step

    Returns:
        (T, 2) predicted positions
    """
    steps = int(horizon / dt)
    t = np.arange(1, steps + 1) * dt
    x = state.x + state.vx * t
    y = state.y + state.vy * t
    return np.column_stack([x, y])


def predict_constant_acceleration(state: VehicleState, horizon: float,
                                  dt: float = 0.1) -> np.ndarray:
    """Predict assuming constant acceleration along current heading.

    Args:
        state: Current vehicle state (uses velocity and acceleration)
        horizon: Prediction horizon (seconds)
        dt: Time step

    Returns:
        (T, 2) predicted positions
    """
    steps = int(horizon / dt)
    t = np.arange(1, steps + 1) * dt

    # Speed evolution: v(t) = v0 + a*t, clamped at 0
    v = np.maximum(state.velocity + state.acceleration * t, 0.0)

    # Distance along heading: integral of v(t)
    # s(t) = v0*t + 0.5*a*t^2
    s = state.velocity * t + 0.5 * state.acceleration * t ** 2
    s = np.maximum(s, 0.0)

    x = state.x + s * np.cos(state.heading)
    y = state.y + s * np.sin(state.heading)
    return np.column_stack([x, y])


def predict_ctra(state: VehicleState, horizon: float,
                 dt: float = 0.1) -> np.ndarray:
    """Constant Turn Rate and Acceleration (CTRA) model.

    More accurate for turning vehicles. Uses yaw_rate.

    Args:
        state: Current vehicle state (uses yaw_rate)
        horizon: Prediction horizon
        dt: Time step

    Returns:
        (T, 2) predicted positions
    """
    steps = int(horizon / dt)
    positions = np.zeros((steps, 2))

    x, y = state.x, state.y
    theta = state.heading
    v = state.velocity
    a = state.acceleration
    omega = state.yaw_rate

    for i in range(steps):
        v_new = max(v + a * dt, 0.0)

        if abs(omega) < 1e-6:
            # Straight line
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
        else:
            # Circular arc
            x += (v_new * np.sin(theta + omega * dt) - v * np.sin(theta)) / omega
            y += (-v_new * np.cos(theta + omega * dt) + v * np.cos(theta)) / omega
            theta += omega * dt

        v = v_new
        positions[i] = [x, y]

    return positions


def predict_agent(agent: Agent, horizon: float = 3.0, dt: float = 0.1) -> np.ndarray:
    """Predict trajectory for an agent, choosing appropriate model.

    Selection logic:
    - If agent already has predicted_trajectory, use it
    - High yaw_rate → CTRA model
    - Has acceleration → constant acceleration
    - Otherwise → constant velocity

    Args:
        agent: Agent with state information
        horizon: Prediction horizon (seconds)
        dt: Time step

    Returns:
        (T, 2) predicted positions
    """
    if agent.predicted_trajectory is not None:
        return agent.predicted_trajectory

    state = agent.state

    # Choose model based on available information
    if abs(state.yaw_rate) > 0.05:  # Significant turning
        return predict_ctra(state, horizon, dt)
    elif abs(state.acceleration) > 0.3:  # Significant acceleration
        return predict_constant_acceleration(state, horizon, dt)
    else:
        return predict_constant_velocity(state, horizon, dt)


def predict_all_agents(agents: list[Agent], horizon: float = 3.0,
                       dt: float = 0.1) -> dict[str, np.ndarray]:
    """Predict trajectories for all agents.

    Args:
        agents: List of detected agents
        horizon: Prediction horizon
        dt: Time step

    Returns:
        Dict mapping agent_id → (T, 2) predicted positions
    """
    predictions = {}
    for agent in agents:
        predictions[agent.state.id] = predict_agent(agent, horizon, dt)
    return predictions

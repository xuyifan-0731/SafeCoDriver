"""Safety performance metrics for evaluation."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SafetyMetrics:
    """Aggregated safety metrics for one episode/scenario."""
    collision_count: int = 0
    collision_rate: float = 0.0          # collisions / total_steps
    mean_ttc: float = float('inf')       # Average time-to-collision (seconds)
    min_ttc: float = float('inf')        # Minimum TTC observed
    safety_distance_violations: int = 0  # Times safety distance was breached
    infeasible_region_entries: int = 0   # Times agent entered unsafe region
    min_harm_triggers: int = 0           # Times minimum harm mode was activated
    total_steps: int = 0
    avg_speed: float = 0.0
    completion_rate: float = 0.0         # Task/route completion rate


def compute_ttc(ego_pos: np.ndarray, ego_vel: np.ndarray,
                other_pos: np.ndarray, other_vel: np.ndarray) -> float:
    """Compute Time-to-Collision between two agents.

    Uses relative velocity approach. Returns inf if no collision.

    Args:
        ego_pos: (2,) ego position
        ego_vel: (2,) ego velocity
        other_pos: (2,) other agent position
        other_vel: (2,) other agent velocity

    Returns:
        TTC in seconds, or inf if no collision trajectory
    """
    rel_pos = other_pos - ego_pos
    rel_vel = other_vel - ego_vel
    rel_speed = np.linalg.norm(rel_vel)

    if rel_speed < 1e-6:
        return float('inf')

    # Project relative position onto relative velocity direction
    approach_speed = -np.dot(rel_pos, rel_vel) / rel_speed
    if approach_speed <= 0:
        return float('inf')  # Moving apart

    distance = np.linalg.norm(rel_pos)
    ttc = distance / approach_speed
    return ttc


def compute_safety_distance(speed: float, reaction_time: float = 1.5,
                            deceleration: float = 6.0) -> float:
    """Compute RSS-style minimum safety distance.

    Args:
        speed: Current speed (m/s)
        reaction_time: Driver/system reaction time (seconds)
        deceleration: Maximum comfortable deceleration (m/s^2)

    Returns:
        Minimum safe following distance (meters)
    """
    # RSS formula: d = v * t_react + v^2 / (2 * a_max)
    return speed * reaction_time + speed ** 2 / (2 * deceleration)


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if a point is inside a polygon using ray casting.

    Args:
        point: (2,) point to check
        polygon: (N, 2) polygon vertices

    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False
    px, py = point
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

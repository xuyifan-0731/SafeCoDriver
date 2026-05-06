from __future__ import annotations
"""Blind spot inference from cooperative perception visibility information.

Uses occlusion geometry to identify hidden regions and infer potential
hidden agents. Corresponds to research plan step 2.2.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union

from ..interface import BlindSpot, Agent, VehicleState, AgentType, PerceptionResult


def compute_occlusion_polygon(observer_pos: np.ndarray,
                              obstacle_corners: np.ndarray,
                              max_range: float = 100.0) -> Optional[np.ndarray]:
    """Compute the shadow/occlusion polygon behind an obstacle from observer's view.

    Args:
        observer_pos: (2,) observer position
        obstacle_corners: (N, 2) corners of the occluding obstacle
        max_range: Maximum sensing range

    Returns:
        (M, 2) polygon vertices of the occluded region, or None
    """
    if len(obstacle_corners) < 2:
        return None

    # Find the two extreme corners as seen from the observer
    angles = np.arctan2(obstacle_corners[:, 1] - observer_pos[1],
                        obstacle_corners[:, 0] - observer_pos[0])

    # Handle angle wrapping
    angle_range = np.max(angles) - np.min(angles)
    if angle_range > np.pi:
        # Wrapping case: shift angles
        angles = np.where(angles < 0, angles + 2 * np.pi, angles)

    idx_min = np.argmin(angles)
    idx_max = np.argmax(angles)

    left_corner = obstacle_corners[idx_min]
    right_corner = obstacle_corners[idx_max]

    # Extend rays from observer through corners to max_range
    def extend_ray(origin, point, distance):
        direction = point - origin
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return point
        return origin + direction / norm * distance

    left_far = extend_ray(observer_pos, left_corner, max_range)
    right_far = extend_ray(observer_pos, right_corner, max_range)

    # Occlusion polygon: left_corner → left_far → right_far → right_corner
    shadow = np.array([left_corner, left_far, right_far, right_corner])
    return shadow


def identify_blind_spots(perception: PerceptionResult,
                         sensing_range: float = 80.0) -> list[BlindSpot]:
    """Identify blind spots from ego perspective using agent occlusion.

    For each large agent (vehicle, truck), compute the shadow region behind it.
    Cooperative perception can reduce blind spots — agents visible to cooperative
    partners but not to ego are already in the agent list with is_visible=True.

    Args:
        perception: Cooperative perception result
        sensing_range: Maximum sensor range

    Returns:
        List of BlindSpot regions
    """
    ego_pos = np.array([perception.ego.x, perception.ego.y])
    blind_spots = []

    for agent in perception.agents:
        # Only large objects create significant occlusion
        # Skip agents that are far away or behind ego
        if agent.state.length < 3.0:
            continue

        # Skip agents far from ego (occlusion too narrow to matter)
        dist = np.sqrt((agent.state.x - ego_pos[0]) ** 2 +
                       (agent.state.y - ego_pos[1]) ** 2)
        if dist > sensing_range * 0.5 or dist < 3.0:
            continue

        # Compute obstacle corners (oriented bounding box)
        corners = _get_vehicle_corners(agent.state)

        shadow = compute_occlusion_polygon(ego_pos, corners, sensing_range)
        if shadow is not None and len(shadow) >= 3:
            blind_spots.append(BlindSpot(
                polygon=shadow,
                occluder_id=agent.state.id,
            ))

    # If visibility_map is provided, use it to refine blind spots
    if perception.blind_spots:
        blind_spots.extend(perception.blind_spots)

    return blind_spots


def infer_hidden_agents(blind_spots: list[BlindSpot],
                        ego_state: VehicleState,
                        conservative: bool = True) -> list[Agent]:
    """Infer potential hidden agents in blind spots (conservative estimation).

    For safety, assume the worst case: each significant blind spot may contain
    a vehicle moving toward the ego vehicle.

    Args:
        blind_spots: Identified blind spot regions
        ego_state: Ego vehicle state
        conservative: If True, place phantom agents in each blind spot

    Returns:
        List of inferred phantom agents with low confidence
    """
    if not conservative:
        return []

    phantom_agents = []
    for i, bs in enumerate(blind_spots):
        if len(bs.polygon) < 3:
            continue

        poly = Polygon(bs.polygon)
        if poly.area < 20.0:  # Too small to hide a vehicle
            continue

        # Only infer phantom for blind spots reasonably close to ego
        centroid = poly.centroid
        if centroid.is_empty:
            continue

        dist_to_ego = np.sqrt((ego_state.x - centroid.x) ** 2 +
                              (ego_state.y - centroid.y) ** 2)
        if dist_to_ego > 50.0:  # Too far to be relevant
            continue

        # Assume worst case: moving toward ego at moderate speed
        dx = ego_state.x - centroid.x
        dy = ego_state.y - centroid.y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-3:
            continue

        heading = np.arctan2(dy, dx)  # Toward ego
        speed = 8.0  # Conservative: moderate urban speed

        phantom = Agent(
            state=VehicleState(
                id=f"phantom_{i}",
                x=centroid.x, y=centroid.y,
                heading=heading,
                velocity=speed,
                vx=speed * np.cos(heading),
                vy=speed * np.sin(heading),
                length=4.5, width=1.8,
                vehicle_type="car",
            ),
            agent_type=AgentType.UNKNOWN,
            is_visible=False,
            confidence=0.3,  # Low confidence — inferred, not observed
            source="blind_spot_inference",
        )
        phantom_agents.append(phantom)

    return phantom_agents


def _get_vehicle_corners(state: VehicleState) -> np.ndarray:
    """Compute 4 corners of a vehicle's oriented bounding box.

    Returns:
        (4, 2) corner positions
    """
    L, W = state.length, state.width
    cos_h, sin_h = np.cos(state.heading), np.sin(state.heading)

    # Corners relative to center: front-left, front-right, rear-right, rear-left
    half_l, half_w = L / 2, W / 2
    corners_local = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ])

    # Rotate by heading
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    corners_global = corners_local @ rotation.T + np.array([state.x, state.y])

    return corners_global

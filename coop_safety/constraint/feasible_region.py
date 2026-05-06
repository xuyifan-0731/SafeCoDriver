from __future__ import annotations
"""Feasible region computation based on vehicle dynamics and right-of-way.

Computes the initial set of physically reachable and legally permissible
states for the ego vehicle. Corresponds to research plan steps 3.1-3.2.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.ops import unary_union

from ..interface import PerceptionResult, VehicleState, LaneInfo, CollisionEvent
from ..perception.dynamics import BicycleModel, get_dynamics_params


class FeasibleRegionComputer:
    """Compute initial feasible region before safety constraint tightening.

    Step 3.1: dynamics + right-of-way → initial feasible polygon
    Step 3.2: subtract RiskEvents collision regions → tightened feasible polygon
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.horizon = cfg.get("horizon", 3.0)
        self.dt = cfg.get("dt", 0.1)
        self.n_samples = cfg.get("n_samples", 16)
        self.lane_buffer = cfg.get("lane_buffer", 1.0)  # Extra margin from lane edge

    def compute_initial(self, ego: VehicleState,
                        lanes: list[LaneInfo]) -> Polygon:
        """Compute initial feasible region from dynamics + road boundaries.

        Args:
            ego: Ego vehicle state
            lanes: Available lane information

        Returns:
            Shapely Polygon of feasible positions
        """
        # Step 3.1a: Reachable set from dynamics
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)
        model = BicycleModel(params)
        state = np.array([ego.x, ego.y, ego.heading, ego.velocity])

        reachable_boundary = model.compute_reachable_set(
            state, self.horizon, self.dt, self.n_samples
        )

        if len(reachable_boundary) < 3:
            # Fallback: circle around ego
            angles = np.linspace(0, 2 * np.pi, 32)
            r = ego.velocity * self.horizon + 5.0
            reachable_boundary = np.column_stack([
                ego.x + r * np.cos(angles),
                ego.y + r * np.sin(angles),
            ])

        reachable_poly = Polygon(reachable_boundary)
        if not reachable_poly.is_valid:
            reachable_poly = reachable_poly.buffer(0)

        # Step 3.1b: Road boundary constraint (if lanes available)
        if lanes:
            road_poly = self._build_road_polygon(lanes)
            if road_poly is not None and road_poly.is_valid:
                # Feasible = reachable ∩ road
                feasible = reachable_poly.intersection(road_poly)
                if isinstance(feasible, (Polygon, MultiPolygon)) and not feasible.is_empty:
                    if isinstance(feasible, MultiPolygon):
                        # Take largest polygon
                        feasible = max(feasible.geoms, key=lambda g: g.area)
                    return feasible

        return reachable_poly

    def tighten_by_events(self, feasible: Polygon,
                          events: list[CollisionEvent],
                          ego_pos: Optional[np.ndarray] = None) -> Polygon:
        """Step 3.2: Tighten feasible region based on collision events.

        Two strategies:
        1. If event region overlaps feasible → subtract directly
        2. If event is AHEAD of ego (in feasible direction) → subtract a
           directional wedge from feasible toward the event, proportional
           to event severity × probability
        """
        for event in events:
            if len(event.spatial_region) < 3:
                continue

            try:
                event_poly = Polygon(event.spatial_region)
                if not event_poly.is_valid:
                    continue

                buffer_size = 2.0 * event.probability
                exclusion = event_poly.buffer(buffer_size)

                # Strategy 1: direct overlap
                if feasible.intersects(exclusion):
                    feasible = feasible.difference(exclusion)
                elif ego_pos is not None:
                    # Strategy 2: directional exclusion
                    # Create a wedge from ego toward event center
                    event_center = np.mean(event.spatial_region, axis=0)
                    direction = event_center - ego_pos
                    dist = np.linalg.norm(direction)
                    if dist < 1.0:
                        continue

                    direction = direction / dist
                    risk_factor = event.severity * event.probability

                    # Only exclude if risk is significant
                    if risk_factor > 0.3:
                        # Wedge: exclude the part of feasible that is in the
                        # direction of the event, with width proportional to risk
                        perp = np.array([-direction[1], direction[0]])
                        wedge_width = 5.0 * risk_factor
                        wedge_length = min(dist, 30.0)

                        wedge_pts = np.array([
                            ego_pos + direction * 3.0 - perp * wedge_width,
                            ego_pos + direction * 3.0 + perp * wedge_width,
                            ego_pos + direction * wedge_length + perp * wedge_width * 1.5,
                            ego_pos + direction * wedge_length - perp * wedge_width * 1.5,
                        ])

                        try:
                            wedge = Polygon(wedge_pts)
                            if wedge.is_valid and feasible.intersects(wedge):
                                feasible = feasible.difference(wedge)
                        except Exception:
                            pass

            except Exception:
                continue

        if isinstance(feasible, MultiPolygon):
            feasible = max(feasible.geoms, key=lambda g: g.area)
        if feasible.is_empty:
            return Polygon()
        return feasible

    def _build_road_polygon(self, lanes: list[LaneInfo]) -> Optional[Polygon]:
        """Build drivable road polygon from lane boundaries."""
        lane_polys = []
        for lane in lanes:
            if lane.lane_type != "driving":
                continue
            if len(lane.left_boundary) < 2 or len(lane.right_boundary) < 2:
                continue
            try:
                # Lane polygon: left boundary forward + right boundary reversed
                boundary = np.vstack([
                    lane.left_boundary,
                    lane.right_boundary[::-1],
                ])
                poly = Polygon(boundary).buffer(self.lane_buffer)
                if poly.is_valid:
                    lane_polys.append(poly)
            except Exception:
                continue

        if lane_polys:
            return unary_union(lane_polys)
        return None

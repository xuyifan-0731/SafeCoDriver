from __future__ import annotations
"""Hierarchical safety constraint tightening.

Progressively tightens the feasible region using three layers of risk:
  RiskEvents → RiskGraph (TTC) → RiskMap (zone exclusion)
Corresponds to research plan steps 3.3-3.4.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.ops import unary_union

from ..interface import (
    RiskRegion, RiskLevel, ConflictEdge, CollisionEvent, VehicleState,
)


class HierarchicalConstraint:
    """Apply three-layer constraint tightening to feasible region.

    Tightening order (coarse to fine, matching plan 3.2→3.3→3.4):
    1. RiskEvents exclusion (step 3.2) — handled by FeasibleRegionComputer
    2. TTC-based tightening from RiskGraph (step 3.3)
    3. Risk zone exclusion from RiskMap (step 3.4)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.ttc_critical = cfg.get("ttc_critical", 3.0)     # seconds
        self.ttc_warning = cfg.get("ttc_warning", 6.0)       # seconds
        self.ttc_buffer_scale = cfg.get("ttc_buffer_scale", 4.0)  # larger exclusion
        self.high_risk_exclude_ratio = cfg.get("high_risk_exclude_ratio", 0.9)
        self.medium_risk_exclude_ratio = cfg.get("medium_risk_exclude_ratio", 0.6)

    def tighten_by_ttc(self, feasible: Polygon,
                       conflict_edges: list[ConflictEdge],
                       ego: VehicleState) -> tuple[Polygon, float]:
        """Step 3.3: TTC-based constraint tightening.

        For each conflict with low TTC, expand exclusion zones around
        the predicted collision points.

        Args:
            feasible: Current feasible region
            conflict_edges: Conflict edges from RiskGraph
            ego: Ego vehicle state

        Returns:
            (tightened polygon, minimum TTC encountered)
        """
        min_ttc = float('inf')

        for edge in conflict_edges:
            if edge.ttc >= self.ttc_warning:
                continue

            min_ttc = min(min_ttc, edge.ttc)

            if edge.collision_point is None:
                continue

            # Buffer size inversely proportional to TTC
            # Lower TTC → larger exclusion zone
            if edge.ttc < self.ttc_critical:
                buffer = self.ttc_buffer_scale * (self.ttc_warning - edge.ttc)
            else:
                buffer = self.ttc_buffer_scale * (self.ttc_warning - edge.ttc) * 0.5

            buffer = max(buffer, 1.0)

            try:
                exclusion = Point(edge.collision_point).buffer(buffer)
                feasible = feasible.difference(exclusion)
            except Exception:
                continue

        feasible = self._clean_polygon(feasible)
        return feasible, min_ttc

    def tighten_by_risk_zones(self, feasible: Polygon,
                              risk_map: list[RiskRegion]) -> Polygon:
        """Step 3.4: Exclude high/medium risk zones from feasible region."""
        sorted_regions = sorted(risk_map, key=lambda r: r.risk_level.value, reverse=True)

        for region in sorted_regions:
            if region.risk_level == RiskLevel.LOW:
                break

            if len(region.polygon) < 3:
                continue

            try:
                region_poly = Polygon(region.polygon)
                if not region_poly.is_valid or not feasible.intersects(region_poly):
                    continue

                if region.risk_level == RiskLevel.HIGH:
                    exclude_fraction = self.high_risk_exclude_ratio
                elif region.risk_level == RiskLevel.MEDIUM:
                    exclude_fraction = self.medium_risk_exclude_ratio
                else:
                    continue

                # Direct exclusion — subtract the risk region from feasible
                exclusion = region_poly.buffer(exclude_fraction * 2.0)
                candidate = feasible.difference(exclusion)
                candidate = self._clean_polygon(candidate)

                # Safety check: keep at least 10% of current area
                if not candidate.is_empty and candidate.area > feasible.area * 0.1:
                    feasible = candidate

            except Exception:
                continue

        return feasible

    def apply_full_hierarchy(self, feasible: Polygon,
                             risk_map: list[RiskRegion],
                             conflict_edges: list[ConflictEdge],
                             ego: VehicleState,
                             agents: list = None,
                             ) -> tuple[Polygon, float, list[str]]:
        """Apply complete hierarchical tightening pipeline.

        Order: Events (already done) → Agent proximity → TTC → Risk zones
        """
        reasoning = []
        initial_area = feasible.area

        # Step 3.2.5: Direct agent proximity exclusion
        # For each agent within range, exclude a speed-dependent danger zone
        if agents:
            for agent in agents:
                a = agent.state if hasattr(agent, 'state') else agent
                dx = a.x - ego.x
                dy = a.y - ego.y
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 50 or dist < 1:
                    continue

                # Speed-dependent danger radius:
                # Static buffer (vehicle size) + dynamic buffer (speed × prediction time)
                speed = max(a.velocity, 1.0)
                static_buffer = max(a.length, 4.0) * 0.6
                dynamic_buffer = speed * 1.2  # ~1.2s of travel distance
                buffer = static_buffer + dynamic_buffer

                # Approaching agents get larger buffer
                rel_vx = ego.vx - a.vx if hasattr(ego, 'vx') else 0
                rel_vy = ego.vy - a.vy if hasattr(ego, 'vy') else 0
                approach_speed = -(dx * rel_vx + dy * rel_vy) / max(dist, 0.1)
                if approach_speed > 0:
                    buffer += approach_speed * 0.8  # Approaching: extra buffer

                try:
                    exclusion = Point(a.x, a.y).buffer(buffer)
                    candidate = feasible.difference(exclusion)
                    candidate = self._clean_polygon(candidate)
                    if not candidate.is_empty and candidate.area > initial_area * 0.03:
                        feasible = candidate
                except:
                    pass

            if feasible.area < initial_area:
                reasoning.append(f"Agent proximity: excluded {initial_area - feasible.area:.1f}m²")

        area_before_ttc = feasible.area

        # Step 3.3: TTC tightening
        feasible, min_ttc = self.tighten_by_ttc(feasible, conflict_edges, ego)
        if feasible.area < area_before_ttc:
            reasoning.append(
                f"TTC tightening: excluded {area_before_ttc - feasible.area:.1f}m² "
                f"(min_ttc={min_ttc:.2f}s)"
            )

        area_before_zones = feasible.area

        # Step 3.4: Risk zone exclusion
        feasible = self.tighten_by_risk_zones(feasible, risk_map)
        if feasible.area < area_before_zones:
            reasoning.append(
                f"Risk zone exclusion: excluded {area_before_zones - feasible.area:.1f}m²"
            )

        if not reasoning:
            reasoning.append("No constraint tightening needed (low risk)")

        return feasible, min_ttc, reasoning

    @staticmethod
    def _clean_polygon(geom) -> Polygon:
        """Clean up geometry result to a single valid Polygon."""
        if isinstance(geom, MultiPolygon):
            if geom.is_empty:
                return Polygon()
            return max(geom.geoms, key=lambda g: g.area)
        if isinstance(geom, Polygon):
            return geom if not geom.is_empty else Polygon()
        return Polygon()

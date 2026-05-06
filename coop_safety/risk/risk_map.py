from __future__ import annotations
"""RiskMap: Spatial region-based risk assessment.

Divides the surrounding space into grid cells/regions, evaluates risk for each
based on traffic density, uncontrolled agents, and predicted occupancy.
Corresponds to research plan step 2.4.
"""

import numpy as np
from typing import Optional
from shapely.geometry import Polygon, box, Point

from ..interface import (
    PerceptionResult, Agent, VehicleState, RiskRegion, RiskLevel,
)
from ..perception.prediction import predict_all_agents


class RiskMapBuilder:
    """Build a spatial risk map from perception results.

    The risk map divides the area around the ego vehicle into grid cells
    and assigns each a risk level based on:
    - Current agent density
    - Predicted future agent density
    - Presence of uncontrolled agents (pedestrians, cyclists)
    - Proximity to blind spots
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.range_x = cfg.get("range_x", 40.0)
        self.range_y = cfg.get("range_y", 20.0)
        self.cell_size = cfg.get("cell_size", 8.0)
        self.prediction_horizon = cfg.get("prediction_horizon", 3.0)
        self.prediction_dt = cfg.get("prediction_dt", 0.5)

        # Lower thresholds tuned for real urban traffic (DAIR-V2X)
        self.high_risk_threshold = cfg.get("high_risk_threshold", 0.35)
        self.medium_risk_threshold = cfg.get("medium_risk_threshold", 0.15)

    def build(self, perception: PerceptionResult,
              blind_spot_polygons: Optional[list[np.ndarray]] = None) -> list[RiskRegion]:
        """Build risk map from perception.

        Args:
            perception: Cooperative perception result
            blind_spot_polygons: Optional list of blind spot polygon vertices

        Returns:
            List of RiskRegion objects
        """
        ego = perception.ego

        # Create grid in ego-centered frame
        cells = self._create_grid(ego)

        # Predict agent trajectories
        predictions = predict_all_agents(
            perception.agents, self.prediction_horizon, self.prediction_dt
        )

        # Classify agents
        uncontrolled_types = {"pedestrian", "cyclist", "bicycle"}

        # Evaluate each cell
        risk_regions = []
        for cell_poly, cell_center in cells:
            shapely_cell = Polygon(cell_poly)
            if not shapely_cell.is_valid:
                continue

            # Count current agents in cell
            current_density = 0
            uncontrolled_count = 0
            for agent in perception.agents:
                agent_point = Point(agent.state.x, agent.state.y)
                if shapely_cell.contains(agent_point):
                    current_density += 1
                    if agent.state.vehicle_type in uncontrolled_types:
                        uncontrolled_count += 1

            # Count predicted future agents in cell (time-weighted)
            future_density = 0
            n_timesteps = int(self.prediction_horizon / self.prediction_dt)
            for agent_id, traj in predictions.items():
                step_interval = max(1, int(self.prediction_dt / 0.1))
                for t_idx in range(0, len(traj), step_interval):
                    if t_idx < len(traj):
                        pt = Point(traj[t_idx])
                        if shapely_cell.contains(pt):
                            # Discount future occupancy by time
                            time_discount = 1.0 / (1.0 + t_idx * 0.1)
                            future_density += time_discount

            # Blind spot contribution
            blind_spot_overlap = 0.0
            if blind_spot_polygons:
                for bs_poly in blind_spot_polygons:
                    try:
                        bs_shapely = Polygon(bs_poly)
                        if bs_shapely.is_valid and shapely_cell.intersects(bs_shapely):
                            overlap = shapely_cell.intersection(bs_shapely).area
                            blind_spot_overlap += overlap / shapely_cell.area
                    except Exception:
                        continue

            # Compute composite risk score [0, 1]
            # Weighted combination of factors
            density_score = min(1.0, (current_density * 0.3 + future_density * 0.15))
            uncontrolled_score = min(1.0, uncontrolled_count * 0.25)
            blind_spot_score = min(1.0, blind_spot_overlap * 0.5)

            risk_score = min(1.0,
                             density_score * 0.4 +
                             uncontrolled_score * 0.3 +
                             blind_spot_score * 0.3)

            # Classify risk level
            if risk_score >= self.high_risk_threshold:
                level = RiskLevel.HIGH
            elif risk_score >= self.medium_risk_threshold:
                level = RiskLevel.MEDIUM
            else:
                level = RiskLevel.LOW

            risk_regions.append(RiskRegion(
                polygon=cell_poly,
                risk_level=level,
                risk_score=risk_score,
                density=current_density + future_density,
                uncontrolled_agents=uncontrolled_count,
            ))

        return risk_regions

    def _create_grid(self, ego: VehicleState) -> list[tuple[np.ndarray, np.ndarray]]:
        """Create ego-centered grid cells.

        Returns list of (polygon_vertices, center) tuples.
        """
        cos_h, sin_h = np.cos(ego.heading), np.sin(ego.heading)
        rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        cells = []
        n_x = int(2 * self.range_x / self.cell_size)
        n_y = int(2 * self.range_y / self.cell_size)

        for i in range(n_x):
            for j in range(n_y):
                # Local coordinates (ego-centered)
                x_local = -self.range_x + i * self.cell_size
                y_local = -self.range_y + j * self.cell_size

                # Cell corners in local frame
                corners_local = np.array([
                    [x_local, y_local],
                    [x_local + self.cell_size, y_local],
                    [x_local + self.cell_size, y_local + self.cell_size],
                    [x_local, y_local + self.cell_size],
                ])
                center_local = np.array([x_local + self.cell_size / 2,
                                         y_local + self.cell_size / 2])

                # Transform to global frame
                corners_global = corners_local @ rotation.T + np.array([ego.x, ego.y])
                center_global = rotation @ center_local + np.array([ego.x, ego.y])

                cells.append((corners_global, center_global))

        return cells


def get_risk_regions_by_level(risk_map: list[RiskRegion],
                              level: RiskLevel) -> list[RiskRegion]:
    """Filter risk regions by level."""
    return [r for r in risk_map if r.risk_level == level]

"""API blueprint for research point 2.2 trusted cooperative perception.

This file is a design scaffold only. It is intentionally not imported by the
current SafeCoDriver runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from coop_safety.interface import PerceptionResult


@dataclass
class CooperativeMessage:
    """One cooperative perception message from another vehicle or RSU."""

    source_id: str
    timestamp: float
    pose_xyh: tuple[float, float, float]
    perception: PerceptionResult
    claimed_confidence: float = 1.0
    diagnostics: dict = field(default_factory=dict)


@dataclass
class OffsetEstimate:
    """Residual SE(2) offset estimated for one information source."""

    dx: float = 0.0
    dy: float = 0.0
    dtheta: float = 0.0
    covariance: np.ndarray = field(default_factory=lambda: np.eye(3))
    residual_before: float = 0.0
    residual_after: float = 0.0
    residual_p95: float = 0.0
    inlier_ratio: float = 0.0
    match_count: int = 0
    correctable_score: float = 0.0


@dataclass
class EvidenceItem:
    """A compact explainable evidence item used in local reports."""

    kind: str
    value: float | dict
    weight: float
    description: str


@dataclass
class EvidenceMessage:
    """Minimal sufficient evidence exchanged among cooperative vehicles."""

    issuer_id: str
    target_id: str
    time_window: tuple[float, float]
    trust_alpha: float
    trust_beta: float
    offset: Optional[OffsetEstimate]
    residual_summary: dict
    action: str
    evidence_hash: str


@dataclass
class SourceAvailabilityReport:
    """Availability decision for one source at one time step."""

    source_id: str
    vehicle_trust: float
    message_usability: float
    correctable_score: float
    recommended_action: str
    offset: Optional[OffsetEstimate]
    evidence_chain: list[EvidenceItem] = field(default_factory=list)


@dataclass
class CalibrationResult:
    """Output of TrustCalibLayer."""

    perception: PerceptionResult
    source_reports: dict[str, SourceAvailabilityReport]
    evidence_to_broadcast: list[EvidenceMessage] = field(default_factory=list)


class TrustCalibLayer:
    """Trusted cooperative perception preprocessor.

    Intended usage:
        calib = trust_layer.calibrate(ego_perception, coop_messages, peer_evidence)
        modified_wp, stats = hybrid.constrain_waypoints(waypoints, calib.perception)
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def calibrate(
        self,
        ego_perception: PerceptionResult,
        coop_messages: list[CooperativeMessage],
        peer_evidence: Optional[list[EvidenceMessage]] = None,
    ) -> CalibrationResult:
        """Return a calibrated PerceptionResult and explainable source reports."""
        raise NotImplementedError("Blueprint only; implement under coop_safety/trust/.")

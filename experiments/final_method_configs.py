"""Single source of truth for final 260520 method configurations."""
from __future__ import annotations

from coop_safety.interface import SafetyConstraintModule
from coop_safety.learned.hybrid_safety import HybridSafetyConstraint
from experiments.methods import RSSOnly
from experiments.methods_new_baselines import MAPSafety, RiskMMSafety, UniE2EV2XSafety
from experiments.run_forced_conflict_and_fa import HybridWithGeometricAND
from experiments.method_variants import (
    HybridGeometryOnly,
    HybridGeometryTTC,
    HybridV1Only,
    HybridWithGeometricANDTTC,
    HybridWithGeometricANDTTCAware,
    HybridWithGeometricANDTTCMinHarm,
    HybridWithGeometricANDTTCRearEscape,
)


FINAL_HYBRID_THRESHOLD = 0.30


def final_method_configs(v1, include_full_safety: bool = False):
    """Return (method_name, factory, uses_coop) for final cross-platform eval."""
    base_kwargs = dict(detector_model=v1, base_margin_visible=2.5, base_margin_invisible=4.0)

    def hybrid_base():
        return HybridSafetyConstraint(detection_threshold=FINAL_HYBRID_THRESHOLD, **base_kwargs)

    configs = [
        ("NoCon-egoonly", lambda: None, False),
        ("NoCon-coop", lambda: None, True),
        ("RSS-coop", lambda: RSSOnly(), True),
        ("UniE2EV2X-coop", lambda: UniE2EV2XSafety(safety_threshold=3.0), True),
        ("MAP-coop", lambda: MAPSafety(min_clearance=0.5), True),
        ("RiskMM-coop", lambda: RiskMMSafety(v_max=20.0), True),
        ("Hybrid-V1Only-thr0.30", lambda: HybridV1Only(hybrid_base()), True),
        ("Hybrid-GeomOnly", lambda: HybridGeometryOnly(hybrid_base()), True),
        ("Hybrid-Geom+TTC", lambda: HybridGeometryTTC(hybrid_base(), ttc_override=3.0), True),
        ("Hybrid-thr0.30", hybrid_base, True),
        ("Hybrid+AND-thr0.30", lambda: HybridWithGeometricAND(hybrid_base()), True),
        ("Hybrid+AND+TTC-thr0.30", lambda: HybridWithGeometricANDTTC(hybrid_base(), ttc_override=3.0), True),
        ("Hybrid+AND+TTC+RearAware-thr0.30",
         lambda: HybridWithGeometricANDTTCAware(hybrid_base(), ttc_override=3.0), True),
        ("Hybrid+AND+TTC+MinHarm-thr0.30",
         lambda: HybridWithGeometricANDTTCMinHarm(
             hybrid_base(), ttc_override=3.0, rear_gap_guard=18.0), True),
        ("Hybrid+AND+TTC+RearEscape-thr0.30",
         lambda: HybridWithGeometricANDTTCRearEscape(
             hybrid_base(), ttc_override=3.0, rear_gap_guard=18.0), True),
    ]

    if include_full_safety:
        configs.insert(6, ("FullSafety-coop", lambda: SafetyConstraintModule(), True))

    return configs

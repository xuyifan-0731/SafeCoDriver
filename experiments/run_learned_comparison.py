"""Experiment comparing rule-based vs learned risk assessment.

Methods:
  - Ours-Rule: rule-based RiskMap + RiskGraph (current)
  - Ours-Learned: learned Risk Assessment Network replacing RiskMap + RiskGraph
  - All baselines unchanged

Usage:
    cd /raid/xuyifan/jiqiuyu && python experiments/run_learned_comparison.py
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from coop_safety.interface import SafetyConstraintModule
from coop_safety.learned.risk_assessor import LearnedRiskAssessor
from experiments.dairv2x_loader_v2 import DAIRv2xLoaderV2
from experiments.methods import NoConstraint, RSSOnly, CBFBased, AblationMethod
from experiments.methods_modern import RiskPotentialField, TTCReachability, SocialForceSafety
from experiments.scenarios import ScenarioConfig
from experiments.run_experiments import evaluate_method_on_scenario

DATA_DIR = "/raid/xuyifan/jiqiuyu/data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure"


class LearnedSafetyModule:
    """SafetyConstraintModule with learned risk assessment replacing rule-based."""

    name = "Ours-Learned"

    def __init__(self):
        self._rule_module = SafetyConstraintModule()
        self._learned = LearnedRiskAssessor()

    def constrain(self, perception):
        """Use learned risk for map+graph, then rule-based constraint pipeline."""
        # Replace risk assessment with learned version
        from coop_safety.interface import ThreeLayerRisk
        from coop_safety.risk.risk_events import RiskEventEnumerator

        # Learned risk map + risk graph
        risk_map = self._learned.assess_risk_map(perception)
        risk_graph = self._learned.assess_conflicts(perception.ego, perception.agents)

        # Rule-based risk events (on top of learned graph)
        agents_by_id = {a.state.id: a for a in perception.agents}
        enumerator = RiskEventEnumerator()
        risk_events = enumerator.enumerate(risk_graph, agents_by_id, perception.ego)

        # Use rule-based constraint pipeline with learned risk inputs
        from coop_safety.constraint.feasible_region import FeasibleRegionComputer
        from coop_safety.constraint.hierarchical import HierarchicalConstraint
        from coop_safety.constraint.feasibility_check import FeasibilityChecker
        from coop_safety.constraint.min_harm import MinimumHarmPlanner
        from coop_safety.interface import SafeActionSpace, ConstraintMode
        from coop_safety.perception.dynamics import get_dynamics_params
        from shapely.geometry import Polygon

        ego = perception.ego
        fc = FeasibleRegionComputer()
        hc = HierarchicalConstraint()
        fcheck = FeasibilityChecker()
        mh = MinimumHarmPlanner()

        reasoning = [f"Learned risk: {sum(1 for r in risk_map if r.risk_level.value >= 2)} HIGH, "
                     f"{len(risk_graph)} conflicts, {len(risk_events)} events"]

        # Constraint pipeline
        feasible = fc.compute_initial(ego, perception.lanes)
        ego_pos = np.array([ego.x, ego.y])
        feasible = fc.tighten_by_events(feasible, risk_events, ego_pos)
        feasible, min_ttc, hier_reasons = hc.apply_full_hierarchy(
            feasible, risk_map, risk_graph, ego, agents=perception.agents)
        reasoning.extend(hier_reasons)

        is_feasible, horizon, freas = fcheck.check(feasible, ego, perception.agents)
        reasoning.extend(freas)

        mode = ConstraintMode.NORMAL
        if not is_feasible or feasible.is_empty:
            # Try relaxation
            relaxed_hc = HierarchicalConstraint({"high_risk_exclude_ratio": 0.3,
                                                  "medium_risk_exclude_ratio": 0.0, "ttc_warning": 8.0})
            feasible2 = fc.compute_initial(ego, perception.lanes)
            feasible2 = fc.tighten_by_events(feasible2, risk_events, ego_pos)
            feasible2, _, _ = relaxed_hc.apply_full_hierarchy(feasible2, risk_map, risk_graph, ego, agents=perception.agents)
            if not feasible2.is_empty and feasible2.area > 5:
                feasible = feasible2
                mode = ConstraintMode.CONSERVATIVE
            else:
                return mh.plan(ego, risk_events)

        coords = np.array(feasible.exterior.coords) if hasattr(feasible, 'exterior') and not feasible.is_empty else np.array([[ego.x, ego.y]])
        params = get_dynamics_params(ego.vehicle_type, ego.length, ego.mass)

        return SafeActionSpace(
            feasible_region=coords,
            max_acceleration=params.max_acceleration,
            min_acceleration=-params.max_deceleration,
            max_steering=params.max_steering_angle,
            max_speed=params.max_speed,
            mode=mode,
            safety_margin_ttc=min_ttc,
            reasoning=reasoning,
            future_feasible=is_feasible,
            future_feasible_horizon=horizon,
        )


def main():
    output_dir = Path(__file__).parent / "results" / f"learned_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LEARNED vs RULE-BASED COMPARISON")
    print("=" * 70)

    loader = DAIRv2xLoaderV2(DATA_DIR)

    methods = {
        "NoConstraint": NoConstraint(),
        "RSS-2017": RSSOnly(),
        "CBF-2017": CBFBased(),
        "APF [Rasekhipour17]": RiskPotentialField(),
        "TTCReach [Pek18]": TTCReachability(),
        "SFS [Helbing95]": SocialForceSafety(),
        "Ours-Rule": SafetyConstraintModule(),
        "Ours-Learned": LearnedSafetyModule(),
        "Ours-NoRiskEvents": AblationMethod("Ours-NoRiskEvents", {"disable_risk_events": True, "min_probability": 999}),
    }
    methods["Ours-Rule"].name = "Ours-Rule"

    indices = np.linspace(0, len(loader) - 1, 500, dtype=int)
    all_metrics = []

    t0 = time.time()
    for i, idx in enumerate(indices):
        if i % 100 == 0:
            print(f"[{i}/500] {time.time()-t0:.0f}s")
        p = loader.load_frame(idx)
        config = ScenarioConfig(name=f"f{idx}", description="", seed=42+i, difficulty="hard")
        for mname, method in methods.items():
            m = evaluate_method_on_scenario(method, p, config)
            all_metrics.append(m)

    # Summary
    print("\n" + "=" * 60)
    method_names = sorted(set(m.method_name for m in all_metrics))
    print(f"{'Method':25s} {'Area':>8s} {'Tight%':>7s} {'TTC':>6s} {'MH':>6s} {'ms':>5s}")
    print("-" * 55)
    rows = []
    for mname in method_names:
        mm = [m for m in all_metrics if m.method_name == mname]
        a = np.mean([m.feasible_area for m in mm])
        t = np.mean([m.constraint_tightening_ratio for m in mm])
        ttc = np.mean([min(float(m.min_ttc) if not isinstance(m.min_ttc, str) else 100, 100) for m in mm])
        mh = sum(1 for m in mm if m.mode == "minimum_harm")
        tm = np.mean([m.compute_time_ms for m in mm])
        print(f"{mname:25s} {a:8.1f} {t:6.1%} {ttc:6.2f} {mh:>3d}/500 {tm:5.0f}")
        rows.append({"method": mname, "area": a, "tightening": t, "ttc": ttc, "mh": mh, "time": tm})

    # Save
    results = {
        "info": {"timestamp": datetime.now().isoformat(), "n_frames": 500, "n_methods": len(methods)},
        "summary": rows,
        "metrics": [{k: (str(v) if isinstance(v, float) and (np.isinf(v) or np.isnan(v)) else v)
                     for k, v in asdict(m).items()} for m in all_metrics],
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    lines = ["# Learned vs Rule-Based Comparison\n"]
    lines.append("| Method | Area | Constraint% | TTC | Min-Harm | Time |")
    lines.append("|--------|------|------------|-----|----------|------|")
    for r in rows:
        lines.append(f"| {r['method']} | {r['area']:.1f} | {r['tightening']:.1%} | {r['ttc']:.2f}s | {r['mh']}/500 | {r['time']:.0f}ms |")
    (output_dir / "summary.md").write_text("\n".join(lines))

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()

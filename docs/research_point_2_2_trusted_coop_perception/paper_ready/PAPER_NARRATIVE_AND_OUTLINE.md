# Paper Narrative And Outline

## Working Title

Safety-Aware Information Usability for Trustworthy Cooperative Perception under Spatial Mismatch and Abnormal Messages

## Central Thesis

Cooperative perception messages should not be treated as simply normal or abnormal. Under spatial mismatch, temporal staleness, dropout, fake objects, and noisy compound anomalies, the key question is whether each message and object is usable, correctable, recoverable, or safety-critical enough to filter. This work frames cooperative perception robustness as safety-aware information-usability estimation.

## Core Contributions

1. **Unified information-usability framework** for cooperative perception under mixed abnormal information sources, including spatial shift, stale messages, dropout, fake-object injection, and noise+fake compound anomalies.
2. **Decoupled message-level and object-level evidence mechanism** that uses vehicle/message calibration, multi-peer object support, box-margin filtering, consensus smoothing, and missing-object recovery.
3. **Safety-aware object admission** that connects perception usability to downstream waypoint collision risk, including path-risk-aware admission for real multi-source labels.
4. **Real cooperative label calibration audit** for DeepAccident, including the correct label/lidar-to-ego transform and target-ego duplicate filtering.
5. **Boundary-aware evaluation** with ablations, support-quality stress, seed robustness, collusion stress, real multi-source diagnostics, scenario bootstrap, and runtime measurements.

## Main Claim Scope

### Strong Synthetic Claim

Under reliable multi-peer evidence, the proposed framework restores clean-level waypoint safety across spatial shift, temporal staleness, dropout, fake-object injection, and noise+fake compound anomalies.

Supported by:

```text
paper_ready/tables/main_synthetic_results.md
paper_ready/tables/baseline_comparison_20x20_v1.md
paper_ready/tables/scenario_bootstrap_synthetic_20x20_v1.md
paper_ready/tables/ablation_results_20x20_fast.md
paper_ready/tables/support_quality_results.md
paper_ready/tables/collusion_stress_20x20_v1.md
```

### Disciplined Real-Data Claim

Real DeepAccident cooperative labels can be calibrated into the ego label frame with near-zero median residual. Real multi-source evidence is directionally beneficial after target-ego duplicate filtering and path-risk-aware admission, but it remains recall-limited and does not close the gap to the CleanCoop oracle.

Supported by:

```text
paper_ready/tables/realcoop_results.md
paper_ready/tables/realmultisource_results.md
paper_ready/tables/realmultisource_pathrisk_results.md
paper_ready/tables/paired_bootstrap_real_multisource.md
```

## Proposed Paper Structure

1. **Introduction**
   - Cooperative perception improves blind-spot reasoning but is vulnerable to mixed abnormal information.
   - Existing work often treats calibration error, faults, and attacks separately.
   - This paper proposes safety-aware information usability as the unifying lens.

2. **Problem Formulation**
   - Multi-source cooperative message model.
   - Information usability: usable, correctable, recoverable, downweighted, quarantined.
   - Downstream safety metric: waypoint collision rate.

3. **Method**
   - TrustCalib: spatial/time correctability and message action.
   - Multi-peer object evidence: object support, trust-weighted support, cluster spread.
   - BoxGuard and path-risk admission: safety-aware object filtering/recovery.
   - MissingRecovery and PeerConsensusSmoothing.
   - Vehicle/message trust update and evidence exchange interpretation.

4. **Experimental Setup**
   - DeepAccident checkpoint validation split.
   - Synthetic abnormal message generation.
   - Real cooperative label alignment and target-ego duplicate filtering.
   - Baselines and ablations.
   - Metrics: WPC, warning rate, fake removal, missing precision/recall, runtime.

5. **Synthetic Results**
   - Main clean-level recovery table.
   - Baseline comparison.
   - Ablation.
   - Scenario bootstrap.
   - Support-quality and collusion boundary.

6. **Real Cooperative Label Results**
   - Alignment audit.
   - Single-sender results.
   - Multi-source path-risk admission.
   - Real-data limitations and recall bottleneck.

7. **Discussion**
   - Why safety-aware admission matters.
   - What is solved vs not solved.
   - Collusion and support-quality boundaries.
   - Deployment limitations.

8. **Conclusion**
   - Information usability is a practical way to handle mixed abnormal cooperative perception.
   - Reliable multi-peer evidence enables clean-level synthetic recovery.
   - Real multi-source evidence requires stronger temporal/source tracking.

## Required Figures

1. **Architecture diagram**: raw messages -> usability estimation -> evidence support -> safety-aware correction/recovery -> SafeCoDriver.
2. **Evidence-chain diagram**: message-level trust, object-level support, path-risk features, final action.
3. **Synthetic qualitative case**: fake-front object removed and waypoint collision avoided (`figures/case_fake_front_filter.png`).
4. **Compound qualitative case**: noisy fake-front message corrected by peer evidence and recovery (`figures/case_noise_fake_recovery.png`).
5. **Missing recovery case**: dropout blind-spot object recovered from peers (`figures/case_drop_missing_recovery.png`).
6. **Collusion boundary case**: trust weighting prevents one low-trust colluding support from validating a fake object (`figures/case_collusion_trust_weighting.png`).
7. **Real path-risk case**: one-source object admitted only because it intersects the planned waypoint corridor (`figures/case_real_pathrisk_admission.png`).
8. **Boundary figure**: support-quality / collusion results showing where the method succeeds and fails.

## Current Weak Points To Address Before Submission

1. Real-data superiority is directional but not statistically strong.
2. Temporal evidence is implemented only as a candidate persistence diagnostic; motion-consistent source-level tracking is still missing.
3. Full-frame scenario-level bootstrap is not yet available for every synthetic ablation.
4. Need a concise mathematical formulation of usability score and final action selection for the method section.
5. The generated qualitative figures should be converted into final paper layout panels with consistent labels/captions during manuscript drafting.

# A-Class Experiment Flow And Current Status

**Date**: 2026-05-30  
**Workspace**: `/raid/xuyifan/trusted_coop_perception`  
**Codebase dependency**: `/raid/xuyifan/jiqiuyu`  
**Conda env**: `Android-Lab`

## Target Standard

To move this work toward an A-class article, the experiments need to support five claims:

1. **Correctness**: abnormal cooperative messages can be detected, corrected, recovered, or filtered under a unified information-usability framework.
2. **Safety relevance**: improvements must reduce waypoint collision rate, not only improve object-list consistency.
3. **Ablation validity**: each module has a measurable role.
4. **Robustness boundary**: support quality, random seeds, fake/colluding support, and real cooperative labels must be tested.
5. **Claim discipline**: synthetic GT-derived support and real single-sender cooperative labels must be separated.

## Implemented Mechanism

Current final synthetic-pilot mechanism:

```text
TrustCalib
+ TimeCalib
+ MultiPeerObjectGuard
+ BoxGuard
+ PeerConsensusSmoothing
+ MissingRecovery
```

Recommended command:

```bash
python prototype/run_deepaccident_multipeer_pilot.py \
  --support-modes clean,shift \
  --enable-time-calib \
  --enable-missing-recovery \
  --missing-recovery-primary-actions accept,correct,time_correct,downweight,quarantine \
  --enable-box-margin-guard \
  --enable-peer-consensus-smoothing
```

## Completed Experiments

### 1. Main Synthetic Perturbation Results

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/main_synthetic_results.md
```

Key outputs:

```text
results/final_consensus_policy_20x20_fast/summary.csv
results/final_consensus_policy_v1_20x20/summary.csv
results/final_consensus_policy_full_val_fast/summary.csv
results/final_consensus_policy_v1_accident_window/summary.csv
```

Main result:

```text
20x20 V1:
  clean / shift / stale / drop / fake_front / shift+drop /
  stale+drop / noise+fake_front all reach WPC = 0.425%.

Full-frame fast:
  all final-method rows reach WPC = 0.291%.

Accident-window V1:
  all final-method rows reach WPC = 0.606%.
```

Interpretation: under reliable GT-derived support peers, the final method recovers clean-level WPC for spatial shift, stale delay, dropout, fake-front injection, and noise+fake-front compound anomalies.

### 2. Ablation Results

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/ablation_results_20x20_fast.md
```

Key outputs:

```text
results/ablation_no_consensus_smoothing_20x20_fast/summary.csv
results/ablation_no_noisy_recovery_20x20_fast/summary.csv
results/ablation_no_boxguard_20x20_fast/summary.csv
results/ablation_no_missing_recovery_20x20_fast/summary.csv
```

Important findings:

| Variant | Critical Observation |
|---|---|
| final | `noise+fake_front` WPC = 0.425% |
| w/o noisy recovery | `noise+fake_front` WPC rises to 0.700% |
| w/o BoxGuard | `noise+fake_front` WPC rises to 2.425%; fake removal collapses |
| w/o MissingRecovery | `drop` WPC rises to 1.300% |
| w/o smoothing | `noise+fake_front` WPC = 0.450%, close to final but not exactly clean |

Interpretation: BoxGuard and MissingRecovery are essential. PeerConsensusSmoothing helps, but the largest noise-compound recovery effect comes from evidence-gated recovery from downweighted/quarantined primary messages.

### 3. Seed Robustness

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/seed_robustness.md
```

Key output:

```text
results/seed_robustness_final_20x20_fast/
```

Seeds:

```text
1, 3, 5, 7, 9
```

Main result:

```text
drop:
  WPC mean = 0.425%, std = 0.000%

noise+fake_front:
  WPC mean = 0.425%, std = 0.000%
  fake removal mean = 98.79%
```

Interpretation: the final 20x20 result is not a seed-specific artifact for stochastic dropout/noise perturbations.

### 4. Support Quality And Boundary

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/support_quality_results.md
```

Key outputs:

```text
results/support_quality_final_clean_noise_20x20_fast/summary.csv
results/support_quality_final_clean_drop_20x20_fast/summary.csv
results/support_quality_final_drop_drop_20x20_fast/summary.csv
```

Important findings:

| Support Modes | drop WPC | noise+fake_front WPC | Interpretation |
|---|---:|---:|---|
| clean+shift | 0.425% | 0.425% | reliable support recovers clean-level |
| clean+noise | 0.875% | 0.450% | noisy support weakens missing recovery |
| clean+drop | 1.150% | 0.775% | dropout support reduces evidence completeness |
| drop+drop | 1.425% | 1.325% | insufficient support is a real failure boundary |

Interpretation: the method should not claim robustness under arbitrarily degraded support. The correct paper claim is evidence-conditioned robustness.

### 5. Real DeepAccident Other-Vehicle Label Alignment

Implemented:

```text
prototype/real_coop.py
prototype/run_deepaccident_realcoop_alignment_audit.py
prototype/run_deepaccident_realcoop_pilot.py
```

Key output:

```text
results/realcoop_alignment_full_val_self_filtered/summary.json
paper_ready/realcoop_alignment_summary_self_filtered.json
```

Alignment result:

```text
frames = 1753
coverage = 100%
ID residual median = 1.53e-5 m
visible ID residual median = 1.40e-5 m
invisible ID residual median = 1.72e-5 m
greedy residual median = 1.51e-5 m
avg real other ego-invisible objects = 12.21/frame
```

Correct transformation:

```text
source label/lidar
-> source lidar_to_ego
-> source ego_to_world
-> inverse(target ego_to_world)
-> inverse(target lidar_to_ego)
-> target label/lidar
```

Input-model correction:

```text
source-label copies of the target ego vehicle are removed after alignment if
they fall within 2.5 m of the ego origin.
```

Interpretation: real `other_vehicle` labels can be aligned accurately. The previous direct-coordinate audit was incomplete because it did not use calibration pickles. A later real-coop diagnostic also found that source labels may include the target ego vehicle as an object; keeping this object is an invalid cooperative-perception input because it turns the ego vehicle into a false obstacle at the ego origin.

### 6. Real Single-Sender Cooperative Pilot

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/realcoop_results.md
```

Key outputs:

```text
results/realcoop_pilot_20x20_fast_self_filtered/summary.csv
results/realcoop_pilot_v1_20x20_self_filtered/summary.csv
results/realcoop_pilot_full_val_fast_self_filtered/summary.csv
```

Important findings:

```text
20x20 V1:
  EgoOnly WPC              = 1.850%
  CleanCoop oracle WPC     = 0.425%
  RealOtherRaw WPC         = 1.725%
  RealOtherTrustCalib WPC  = 1.725%
  RealOtherObjectGuard WPC = 1.850%

Full-frame fast:
  EgoOnly WPC              = 1.888%
  CleanCoop oracle WPC     = 0.291%
  RealOtherRaw WPC         = 1.580%
  RealOtherTrustCalib WPC  = 1.580%
  RealOtherObjectGuard WPC = 1.888%
```

Interpretation: after removing target-ego duplicates, real single-sender cooperative labels are usable after alignment and modestly improve raw WPC. However, one sender alone is not enough evidence for oracle-level blind-spot recovery. The conservative single-sender ObjectGuard avoids admitting one-sender high-impact objects and therefore returns to EgoOnly-level WPC.

### 7. Real Multi-Source Cooperative Pilot

Implemented:

```text
prototype/run_deepaccident_realmultisource_pilot.py
```

Paper-ready table:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/realmultisource_results.md
```

Key outputs:

```text
results/realmultisource_20x20_fast_min2_self_filtered_diag/summary.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/frame_diagnostics.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/cluster_records.csv
```

Important findings:

```text
20x20 min2 baseline, source roles =
  other_vehicle, infrastructure, ego_vehicle_behind, other_vehicle_behind

EgoOnly WPC                  = 1.850%
CleanCoop oracle WPC         = 0.425%
RealPrimaryRaw WPC           = 1.725%
RealPrimaryTrustCalib WPC    = 1.725%
RealMultiEvidenceGuard WPC   = 1.625%

RealMultiEvidenceGuard:
  avg missing recovered/frame = 1.20
  missing precision           = 98.75%
  missing recall              = 9.12%
  worse than EgoOnly          = 6/400 frames
  better than EgoOnly         = 12/400 frames
```

Path-risk-aware admission update:

```text
20x20 pathrisk-thr0:
  RealMultiEvidenceGuard WPC   = 1.550%
  missing precision            = 98.76%
  missing recall               = 9.17%
  path-risk single-source TP/FP = 3/0

20x20 V1 pathrisk-thr0:
  RealMultiEvidenceGuard WPC   = 1.550%

full-fast pathrisk-thr0:
  EgoOnly WPC                  = 1.888%
  RealPrimaryTrustCalib WPC    = 1.580%
  RealMultiEvidenceGuard WPC   = 1.563%
```

Interpretation: real multi-source evidence is now beneficial but still far from the oracle. The remaining bottleneck is recall, not precision. Path-risk-aware admission gives a useful refinement: it matches unconstrained single-source WPC improvement while avoiding the precision collapse caused by admitting all one-source candidates. The optional temporal path-risk variant recovered additional true objects but did not further reduce WPC on this subset, so it remains a diagnostic mechanism rather than the recommended policy.

### 8. Statistical Intervals And Runtime

Paper-ready tables:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/statistical_intervals.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/paired_bootstrap_real_multisource.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/runtime_results.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/realmultisource_pathrisk_results.md
```

Key statistical check:

```text
synthetic full-val final:
  clean / drop / fake_front / noise+fake_front all have
  WPC = 51/17530 = 0.291%, Wilson 95% CI [0.221%, 0.382%]

real multi-source 20x20:
  RealPrimaryTrustCalib WPC       = 69/4000 = 1.725%
  min2 RealMultiEvidenceGuard WPC = 65/4000 = 1.625%
  pathrisk RealMultiEvidenceGuard = 62/4000 = 1.550%
  pathrisk scenario bootstrap diff vs EgoOnly = -0.300 pp,
  95% CI [-1.175 pp, 0.400 pp]
```

Runtime probes:

```text
synthetic final 20x20, two modes: 41.46 s for 800 mode-frames
real single-sender 20x20:       6.87 s for 400 frames
real multi-source 20x20:        11.19 s for 400 frames
```

Interpretation: the statistical interval table supports the strong synthetic claim. The real multi-source improvement is directionally positive but should be framed as a diagnostic result because the confidence intervals overlap and scenario bootstrap does not yet show a strong real-data superiority claim.

### 9. Unified Baselines, Scenario Bootstrap, And Collusion Stress

Paper-ready tables:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/baseline_comparison_20x20_v1.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/scenario_bootstrap_synthetic_20x20_v1.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/collusion_stress_20x20_v1.md
/raid/xuyifan/trusted_coop_perception/paper_ready/tables/collusion_bootstrap_20x20_v1.md
```

Baseline comparison on 20x20 V1:

```text
drop:
  Raw fusion / TrustCalib only / w/o MissingRecovery = 1.300% WPC
  Final                                           = 0.425% WPC

fake_front:
  Raw fusion / TrustCalib only = 2.175% WPC
  Final                       = 0.425% WPC

noise+fake_front:
  Raw fusion                  = 1.925% WPC
  TrustCalib only             = 2.925% WPC
  w/o BoxGuard                = 2.425% WPC
  w/o MissingRecovery         = 0.700% WPC
  w/o smoothing               = 0.450% WPC
  Final                       = 0.425% WPC
```

Scenario bootstrap highlights:

```text
noise+fake_front:
  Final - TrustCalib only = -2.500 pp, 95% CI [-4.025, -1.250]
  Final - w/o BoxGuard    = -2.000 pp, 95% CI [-3.175, -1.075]

fake_front:
  Final - TrustCalib only = -1.750 pp, 95% CI [-3.551, -0.400]
```

Collusion stress:

```text
support_modes = clean,fake_front

count-min1:
  fake_front WPC       = 1.800%
  noise+fake_front WPC = 2.500%

count-min2:
  fake_front WPC       = 0.425%
  noise+fake_front WPC = 0.450%

trust-weighted:
  fake_front WPC       = 0.425%
  noise+fake_front WPC = 0.425%
```

Interpretation: calibration-only and raw-fusion baselines fail under dropout/fake/noise-compound anomalies. The final method's largest statistically stable gains come from fake-object filtering and BoxGuard under noise+fake. Collusion stress establishes a clear boundary: count-only single-support evidence is vulnerable, while a stricter support threshold or low trust weight for the colluding sender restores safe behavior.

### 10. Paper Narrative And Figure Scaffolds

Added:

```text
/raid/xuyifan/trusted_coop_perception/paper_ready/PAPER_NARRATIVE_AND_OUTLINE.md
/raid/xuyifan/trusted_coop_perception/paper_ready/figures/architecture.mmd
/raid/xuyifan/trusted_coop_perception/paper_ready/figures/evidence_decision_flow.mmd
```

Narrative consolidation:

```text
Safety-aware information usability is the central concept.
The method should be presented as a unified message/object/path-risk decision
framework rather than a list of independent modules.
```

## Claim Boundary For Paper

Safe main claim:

```text
Under reliable multi-peer evidence, the proposed information-usability
framework restores clean-level waypoint safety across spatial shift, temporal
staleness, dropout, fake-object injection, and noise+fake compound anomalies.
```

Safe real-data claim:

```text
Real DeepAccident other_vehicle labels can be calibrated into the ego label
frame with near-zero median residual. After filtering target-ego duplicates,
real single-sender labels modestly improve raw WPC but remain far from the
CleanCoop oracle. Real multi-source evidence with evidence-gated and
path-risk-aware recovery improves WPC further, but its recall remains low,
establishing the need for stronger temporal evidence and real object admission.
```

Avoid claiming:

```text
full real multi-vehicle validation
robustness to arbitrary support degradation
robustness to fully colluding fake support without trust/evidence penalties
end-to-end production-ready V2X perception
```

## Remaining Work Toward A-Class Submission

These are the next execution items, in priority order:

1. **Stronger temporal evidence for real sources**: the first temporal path-risk variant is implemented, but it did not improve WPC beyond path-risk admission. Next step is source-level track management with motion consistency and persistence scores.
2. **Full scenario-level paired bootstrap**: extend the current synthetic baseline/bootstrap table to full-frame runs when frame-level logs are available.
3. **Qualitative case figures**: generate concrete case visualizations for fake-front filtering, missing-object recovery, collusion failure/recovery, and real path-risk admission.
4. **Method math cleanup**: write the final usability/action-selection equations so the method is reviewable as a compact framework.

## Current Paper-Ready Artifact Index

```text
paper_ready/tables/main_synthetic_results.md
paper_ready/tables/ablation_results_20x20_fast.md
paper_ready/tables/seed_robustness.md
paper_ready/tables/support_quality_results.md
paper_ready/tables/realcoop_results.md
paper_ready/tables/realmultisource_results.md
paper_ready/tables/realmultisource_pathrisk_results.md
paper_ready/tables/statistical_intervals.md
paper_ready/tables/paired_bootstrap_real_multisource.md
paper_ready/tables/runtime_results.md
paper_ready/tables/baseline_comparison_20x20_v1.md
paper_ready/tables/scenario_bootstrap_synthetic_20x20_v1.md
paper_ready/tables/collusion_stress_20x20_v1.md
paper_ready/tables/collusion_bootstrap_20x20_v1.md
paper_ready/PAPER_NARRATIVE_AND_OUTLINE.md
paper_ready/figures/architecture.mmd
paper_ready/figures/evidence_decision_flow.mmd
paper_ready/realcoop_alignment_summary_self_filtered.json
results/CONSENSUS_SMOOTHING_FINAL_UPDATE_2026-05-28.md
```

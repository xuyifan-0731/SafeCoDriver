# Automation Status

**Last update**: 2026-05-28 12:45 UTC

## Completed

- Cloned ARIS reference repository to:

```text
/raid/xuyifan/trusted_coop_perception/aris_repo
```

- Created persistent research workspace:

```text
/raid/xuyifan/trusted_coop_perception/research
```

- Added reproducible literature search runner:

```text
/raid/xuyifan/trusted_coop_perception/scripts/run_lit_search.sh
```

- Ran the first literature search batch. Successful raw outputs:

```text
research/01_literature/raw/arxiv_pose_error.json
research/01_literature/raw/arxiv_attack.json
research/01_literature/raw/arxiv_trust.json
research/01_literature/raw/arxiv_byzantine_fusion.json
research/01_literature/raw/arxiv_faults.json
research/01_literature/raw/openalex_robust_v2x.json
```

- Built a deduplicated paper table:

```text
research/01_literature/PAPER_TABLE.csv
research/01_literature/PAPER_TABLE.md
```

- Current unique candidate papers: **68** after adding trust/misbehavior-specific queries.

- Added literature triage:

```text
research/01_literature/TRIAGE.md
```

- Added closest-prior-work alignment:

```text
research/01_literature/CLOSEST_PRIOR_WORK.md
```

- Downloaded and extracted PDF text for 7 closest papers:

```text
research/01_literature/pdfs/
research/01_literature/text/
```

- Added PDF-level deep-read notes:

```text
research/01_literature/DEEP_READ_NOTES.md
```

- Added method constraints after prior-work check:

```text
research/02_ideas/METHOD_CONSTRAINTS_AFTER_PRIOR.md
```

- Implemented minimal TrustCalib prototype:

```text
prototype/trust_calib.py
prototype/run_deepaccident_shift_pilot.py
```

- Ran DeepAccident shift pilots:

```text
results/deepaccident_shift_smoke/
results/deepaccident_shift_pilot_20x20_fast/
results/deepaccident_shift_pilot_20x20_shift4_fast/
results/PILOT_SUMMARY.md
```

Pilot result: TrustCalib matched OracleCalib WPC on 20 scenarios x 20 frames under both medium and severe translation shifts.

- Implemented mixed-anomaly injection and evaluation:

```text
prototype/anomaly_injection.py
prototype/run_deepaccident_mixed_pilot.py
```

- Ran mixed-anomaly pilots:

```text
results/deepaccident_mixed_smoke/
results/deepaccident_mixed_pilot_20x20_fast/
results/deepaccident_mixed_impact_guard_20x20_fast/
results/deepaccident_mixed_guard_all_20x20_fast/
results/MIXED_ANOMALY_SUMMARY.md
```

Key result: TrustCalib solves stable spatial shifts, but fake_front shows that clean visible anchors are not enough to detect a high-impact bogus cooperative-only object. Message-level ImpactGuard mitigates fake_front but damages clean cooperative perception, so the next variant must use object-level impact attribution.

- Implemented ObjectGuard and added object-level metrics:

```text
prototype/run_deepaccident_mixed_pilot.py
results/deepaccident_object_guard_metrics_20x20_fast/
results/OBJECT_GUARD_SUMMARY.md
```

Key result on 20 validation scenarios x 20 frames with V1 disabled:

```text
clean:        ObjectGuard WPC 0.43%, same as CleanCoop
shift:        ObjectGuard WPC 0.43%, corrects 99%, avg offset dx=-2.00, dy=-1.00
shift_severe: ObjectGuard WPC 0.43%, corrects 97%, avg offset dx=-3.96, dy=-1.97
fake_front:   Raw/TrustCalib WPC 2.17%; ObjectGuard WPC 0.40%
fake_front:   fake_removed=379/400, fake_removal_rate=94.75%, avg_obj_removed=0.9475/frame
```

Interpretation: object-level impact quarantine preserves clean cooperative benefit while removing high-impact unsupported cooperative-only objects. This is a stronger research direction than whole-message trust/quarantine.

- Added exploratory temporal evidence mode:

```text
prototype/run_deepaccident_mixed_pilot.py --evidence-mode temporal
results/deepaccident_object_guard_temporal_smoke/
results/TEMPORAL_EVIDENCE_SMOKE.md
```

Smoke result: temporal-only support removes fake objects but also removes clean high-impact cooperative-only objects before history exists. This is a negative result; single-vehicle temporal continuity is insufficient as the sole object evidence source.

- Added `peer_oracle` evidence mode:

```text
prototype/run_deepaccident_mixed_pilot.py --evidence-mode peer_oracle
results/deepaccident_object_guard_peer_smoke/
results/deepaccident_object_guard_peer_20x20_fast/
```

Key result on 20 validation scenarios x 20 frames with V1 disabled:

```text
clean:        ObjectGuard WPC 0.43%, accepts 100%, removes 0 objects/frame
shift:        ObjectGuard WPC 0.43%, corrects 99%
shift_severe: ObjectGuard WPC 0.43%, corrects 97%
fake_front:   ObjectGuard WPC 0.40%, fake_removed=379/400, fake_removal_rate=94.75%
```

Interpretation: peer evidence reproduces the synthetic ObjectGuard result without directly checking the fake object ID/source. This is now the preferred direction for the next prototype.

- Implemented explicit multi-peer evidence pilot:

```text
prototype/run_deepaccident_multipeer_pilot.py
results/deepaccident_multipeer_smoke/
results/deepaccident_multipeer_20x20_fast/
results/deepaccident_multipeer_min2_20x20_fast/
results/deepaccident_multipeer_collude_min1_20x20_fast/
results/deepaccident_multipeer_collude_min2_20x20_fast/
results/deepaccident_multipeer_support_metrics_20x20_fast/
results/deepaccident_multipeer_collude_min1_support_metrics_20x20_fast/
results/deepaccident_multipeer_collude_min2_support_metrics_20x20_fast/
results/deepaccident_multipeer_v1_smoke/
results/deepaccident_multipeer_v1_metrics_smoke/
results/deepaccident_multipeer_v1_20x20/
results/deepaccident_cluster_metrics_smoke/
results/deepaccident_cluster_metrics_20x20_fast/
results/deepaccident_cluster_collude_unweighted_20x20_fast/
results/deepaccident_cluster_collude_trustweighted_20x20_fast/
results/deepaccident_cluster_records_smoke/
results/deepaccident_cluster_records_fake_front_20x20_fast/
results/deepaccident_temporal_probation_smoke/
results/deepaccident_probation_drop_default_20x20_fast/
results/deepaccident_probation_drop_hightrust_safe_20x20_fast/
results/deepaccident_probation_drop_hightrust_override_20x20_fast/
results/deepaccident_probation_fake_hightrust_safe_20x20_fast/
results/deepaccident_probation_fake_hightrust_long_20x20_fast/
results/deepaccident_trust_dynamics_fake_hightrust_20x20_fast/
results/deepaccident_trust_offset_smoke/
results/deepaccident_trust_offset_noise_stale_5x10_fast/
results/deepaccident_trust_offset_shift_severe_5x10_fast/
results/deepaccident_final_default_50x20_fast/
results/deepaccident_final_default_v1_20x20/
results/MULTI_PEER_OBJECT_GUARD_SUMMARY.md
results/CLUSTER_EVIDENCE_SUMMARY.md
results/TEMPORAL_PROBATION_SUMMARY.md
results/TRUST_DYNAMICS_SUMMARY.md
results/FINAL_DEFAULT_COMBINED_SUMMARY.md
research/02_ideas/TRUST_UPDATE_FORMULATION.md
```

Key result on 20 validation scenarios x 20 frames with V1 disabled and support peers `clean,shift`:

```text
clean:        MultiPeerObjectGuard WPC 0.43%, real false removal 0.000/frame
shift:        MultiPeerObjectGuard WPC 0.43%, primary correction 99%
shift_severe: MultiPeerObjectGuard WPC 0.43%, primary correction 97%
fake_front:   PrimaryTrustCalib WPC 2.17%; MultiPeerObjectGuard WPC 0.40%
fake_front:   fake_removed=94.8%, real false removal 0.000/frame
evidence:     real objects avg peer support 2.0; fake object avg peer support 0.0
```

Colluding support peer stress:

```text
support_modes=clean,fake_front, min_peer_support=1 -> fake removal 0.0%, WPC 2.17%
support_modes=clean,fake_front, min_peer_support=2 -> fake removal 94.8%, WPC 0.40%
collude evidence: real support 2.0, fake support 1.0
```

Interpretation: explicit peer-support threshold is a key design parameter for balancing information availability and resilience to colluding false evidence.

V1-enabled 20x20 check:

```text
clean/CleanCoop:                 WPC 0.43%, warn 56.50%, avg_p_coll 0.228
fake_front/PrimaryTrustCalib:    WPC 2.17%, warn 100.00%, avg_p_coll 0.226
fake_front/MultiPeerObjectGuard: WPC 0.40%, warn 56.50%, avg_p_coll 0.228
```

Interpretation: on this subset, ObjectGuard removes the fake-front disturbance and restores warning behavior to the clean cooperative baseline with V1 enabled.

Cluster evidence update:

```text
cluster metrics added: support_count, trust_weighted_support, position_cov_trace, offset_spread
main cluster run: real support 2.0, fake support 0.0, real false removal 0.000/frame
trust-weighted collusion: real trust support 1.2, fake trust support 0.2
count-only collusion: fake passes, WPC 2.17%
trust threshold min_trust_support=1.0: fake removed 94.8%, WPC 0.40%
```

Interpretation: source trust can now be connected directly to object-level evidence availability rather than only whole-vehicle filtering.

Per-object cluster records:

```text
cluster_records.csv fields:
mode, scenario_index, frame_index, object_id, is_fake, x, y,
support_count, supporting_sender_ids, trust_weighted_support,
position_cov_trace, offset_spread, distance, ttc_s, closest_distance,
geom_delta, mod_delta, final_action
```

Fake-front 20x20 record aggregate:

```text
real cooperative objects: records=7060, quarantined=0, avg_distance=81.374, avg_geom_delta=0.037, avg_mod_delta=0.004
fake_front objects:       records=400, quarantined=379, avg_distance=3.925, avg_closest_distance=1.635, avg_geom_delta=8.107, avg_mod_delta=0.811
```

Path margin added to cluster records:

```text
fields: path_min_distance, path_collision_margin, path_risk_step
fake_front aggregate: avg_path_min_distance=0.009, avg_path_collision_margin=-1.991
real object aggregate: avg_path_min_distance=77.531, avg_path_collision_margin=75.531
```

Temporal/probation update:

```text
cluster records now include primary_sender, evidence_supported, temporal_status, unsupported_age
fake repetition: new_unsupported -> persistent_unsupported, but remains object_quarantine
fake high-trust safe probation: WPC 0.40%, fake removal 94.8%, probation restore 0.0%
drop support default: WPC 1.27%, real false removal 0.065/frame
drop support safe high-trust probation: WPC 1.27%, real false removal 0.065/frame
drop support high-impact override: WPC 0.43%, real false removal 0.000/frame, probation restore 6.5%
```

Interpretation: temporal state should audit unsupported object persistence and drive probation policy, but it should not become evidence by itself. The safe default keeps high-impact unsupported objects quarantined even from high-trust senders; the override is an availability-first ablation.

Trust dynamics update:

```text
enabled primary trust updates from object-level evidence:
  high-impact unsupported object -> trust penalty
  no high-impact unsupported and supported objects exist -> small trust reward
fake_front with high-impact probation override + trust dynamics:
  PrimaryTrustCalib WPC 2.17%, warn 100.00%
  MultiPeerObjectGuard WPC 0.33%, warn 52.00%, fake removal 85.2%, probation restore 9.5%
example trust trajectory:
  frame 0: trust 1.0 -> 0.8, probation restore
  frame 1: trust 0.8 -> 0.6, probation restore
  frame 2+: trust below threshold, object quarantine
```

Interpretation: vehicle trust can grant temporary probation, but repeated high-impact unsupported claims reduce trust and remove that privilege.

Trust update formulation drafted:

```text
research/02_ideas/TRUST_UPDATE_FORMULATION.md
availability = vehicle trust + message correctability + object evidence support + downstream safety impact + temporal probation state
```

Offset-instability trust penalty check:

```text
cluster records now include:
  primary_residual_after, primary_correctable_score,
  frame_offset_instability, frame_peer_disagreement

shift_severe: avg_residual_after=0.054, avg_peer_disagreement=0.000, trust remains 1.0
noise:        avg_residual_after=1.056, avg_peer_disagreement=0.810, trust decays to 0.0
stale:        avg_residual_after=1.227, avg_peer_disagreement=0.449, trust decays to 0.0
```

Final default combined table:

```text
50x20 fast:
  clean:        MultiPeerObjectGuard WPC 0.39%, same as CleanCoop
  shift:        0.66% -> 0.39%, correction 99.1%
  shift_severe: 0.80% -> 0.39%, correction 97.3%
  fake_front:   2.05% -> 0.43%, fake removal 93.6%
  noise/drop/stale remain open

20x20 V1-enabled:
  clean/CleanCoop:                 WPC 0.43%, warn 56.50%, avg_p 0.228
  fake_front/PrimaryTrustCalib:    WPC 2.17%, warn 100.00%, avg_p 0.226
  fake_front/MultiPeerObjectGuard: WPC 0.40%, warn 56.50%, avg_p 0.228
```

Missing-object recovery update:

```text
implemented optional peer-supported missing-object recovery:
  --enable-missing-recovery
  --missing-min-peer-support 2
  --missing-min-trust-support 1.0
  --missing-recovery-primary-actions accept

20x20 fast:
  drop: PrimaryTrustCalib WPC 1.30% -> MultiPeerObjectGuard WPC 0.43%
  missing recovery audit: 4665 recovered records, all in drop mode
  clean/fake_front/noise/stale: 0 missing-recovery records

50x20 fast core:
  drop: PrimaryTrustCalib WPC 1.18% -> MultiPeerObjectGuard WPC 0.39%
  fake_front remains controlled: 2.05% -> 0.43%, fake removal 93.6%

V1 20x20:
  drop: PrimaryTrustCalib WPC 1.30%, warn 53.25%
  drop: MultiPeerObjectGuard WPC 0.43%, warn 57.00%
```

Summary written:

```text
results/MISSING_OBJECT_RECOVERY_SUMMARY.md
results/deepaccident_missing_recovery_20x20_fast/
results/deepaccident_missing_recovery_50x20_fast/
results/deepaccident_missing_recovery_v1_20x20/
```

Oriented-box path margin update:

```text
cluster records now include:
  path_box_distance
  path_box_collision_margin
  path_box_risk_step

20x20 fake_front+drop audit:
  real_primary:      9174 records, negative box margin 0.5%, avg box margin 74.779
  fake_front:         400 records, negative box margin 100.0%, avg box margin -1.900
  missing_recovery:  4665 records, negative box margin 0.5%, avg box margin 74.470
```

Summary written:

```text
results/ORIENTED_BOX_PATH_MARGIN_SUMMARY.md
results/deepaccident_oriented_box_margin_20x20_fast/
```

Box-margin guard ablation:

```text
implemented optional decision feature:
  --enable-box-margin-guard
  --box-margin-guard-thr 0.0

V1-disabled 20x20:
  fake_front: fake removal 100.0%, WPC 0.43%, real removal 0.000/frame
  drop:       WPC 0.43%, missing recovery 11.66/frame
  clean:      WPC 0.43%, real removal 0.000/frame

V1-enabled 20x20:
  fake_front/PrimaryTrustCalib: WPC 2.17%, warn 100.00%
  fake_front/BoxGuard:          WPC 0.43%, warn 56.50%
  drop/BoxGuard:                WPC 0.43%, warn 57.00%
```

Outputs:

```text
results/deepaccident_box_margin_guard_20x20_fast/
results/deepaccident_box_margin_guard_v1_20x20/
```

Experiment audit:

```text
results/EXPERIMENT_AUDIT_2026-05-28.md
```

Key audit findings:

```text
"50x20" labels are misleading: checkpoint val_scenario_idx has 22 scenarios, so those runs are 440 frames.
support_modes=clean,shift is synthetic GT-derived peer evidence, not independent real vehicles.
missing recovery depends on reliable support: support_modes=drop,drop leaves WPC at 1.40%.
BoxGuard with min_peer_support=1 is not collusion-robust: clean+fake_front support leaves WPC at 2.17%.
BoxGuard with min_peer_support=2 recovers fake_front WPC to 0.43%.
```

Corrective reruns and sweeps:

```text
script now writes raw wp_coll/wp_total and actual_scenarios into summary.csv
corrected outputs:
  results/deepaccident_final_default_22x20_fast/
  results/deepaccident_missing_recovery_22x20_fast/
  results/CORRECTED_RESULTS_AND_SWEEPS.md

drop support-quality sweep:
  clean+shift: WPC 0.425%, missing recovery 11.66/frame
  clean+noise: WPC 0.900%, missing recovery 6.92/frame
  clean+drop:  WPC 1.150%, missing recovery 3.51/frame
  drop+drop:   WPC 1.400%, missing recovery 1.07/frame

fake-front collusion sweep:
  clean+shift, min1:             WPC 0.425%, fake removal 100.0%
  clean+fake_front, min1:        WPC 2.175%, fake removal 0.0%
  clean+fake_front, min2:        WPC 0.425%, fake removal 100.0%
  fake_front+fake_front, min2:   WPC 2.175%, fake removal 0.0%
  trust-weighted clean+fake:     WPC 0.425%, fake removal 100.0%
```

Missing-recovery precision/recall:

```text
script now writes:
  avg_missing_gt
  missing_recovery_tp
  missing_recovery_fp
  missing_recovery_precision
  missing_recovery_recall

support-quality sweep:
  clean+shift: precision 1.000, recall 0.967
  clean+noise: precision 1.000, recall 0.580
  clean+drop:  precision 1.000, recall 0.294
  drop+drop:   precision 1.000, recall 0.090

unsafe min1 check:
  support_modes=clean,fake_front, missing_min_peer_support=1
  TP 4704, FP 400, precision 0.922, recall 0.975, WPC 2.175%
```

Summary written:

```text
results/RECOVERY_PRECISION_RECALL_SUMMARY.md
```

Time calibration and final policy:

```text
implemented TimeCalib:
  --enable-time-calib
  --time-delay-grid 0.0,0.25,0.5,0.75,1.0,1.25,1.5

20x20 fast:
  stale: WPC 0.425%, time_correct_rate 99.0%, avg_delay 0.990s
  clean/shift/drop/fake_front: no harmful time-calib trigger

compound anomalies with corrected-message recovery:
  shift+drop: WPC 0.425%, missing recovery 11.66/frame
  stale+drop: WPC 0.425%, missing recovery 11.67/frame
  shift+fake_front: WPC 0.525%, fake removal 93.7%
  noise+fake_front: WPC 0.725%, fake removal 99.4%

V1 20x20 final geometric policy:
  clean/shift/stale/drop/fake_front/shift+drop/stale+drop: WPC 0.425%
  noise+fake_front: WPC 0.725%

full-frame fast over 22 val scenarios:
  frames 1753, wp_total 17530
  clean/drop/fake_front/shift+drop/stale+drop: WPC 0.291%
  stale: WPC 0.297%
  noise+fake_front: WPC 0.713%
```

Summary written:

```text
results/FINAL_METHOD_CONFIGURATION_AND_RESULTS.md
```

## Warnings

- Semantic Scholar returned HTTP 429 rate limit during the first run. The script logs the failure and continues. Re-run later or configure an API key if needed.
- A new red-flag seed appeared: `V2X Misbehavior and Collective Perception Service: Considerations for Standardization`. This increases novelty risk for generic trust/evidence-exchange claims.

## Next Automatic Step

1. Add covariance-aware smoothing/fusion for noise.
2. Add real `other_vehicle` cooperative labels instead of only GT-derived synthetic peers.
3. Package final paper tables from `FINAL_METHOD_CONFIGURATION_AND_RESULTS.md`.

## 2026-05-28 Consensus Smoothing Update

Implemented and validated `--enable-peer-consensus-smoothing` in:

```text
/raid/xuyifan/trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py
```

Final recommended command now includes:

```text
--enable-peer-consensus-smoothing
--missing-recovery-primary-actions accept,correct,time_correct,downweight,quarantine
```

Key outputs:

```text
results/final_consensus_policy_20x20_fast/summary.csv
results/final_consensus_policy_v1_20x20/summary.csv
results/final_consensus_policy_full_val_fast/summary.csv
results/final_consensus_policy_v1_accident_window/summary.csv
results/real_coop_availability_audit/summary.json
results/CONSENSUS_SMOOTHING_FINAL_UPDATE_2026-05-28.md
```

Main finding:

```text
noise+fake_front is no longer an elevated-WPC boundary under reliable
GT-derived support peers:
  20x20 V1:       0.425% WPC, equal to clean
  full-frame fast 0.291% WPC, equal to clean
  accident-window 0.606% WPC, equal to clean
```

Real-coop audit:

```text
DeepAccident other_vehicle labels cover 1753/1753 val frames, but direct
ego-vs-other label residual is ~20.95 m mean / 18.64 m median under the current
loader. These labels need coordinate transformation before being used as
ego-frame real cooperative messages.
```

## 2026-05-30 Real-Coop Correction And Multi-Source Update

Implemented calibrated real-coop alignment and corrected the real-source input model in:

```text
prototype/real_coop.py
prototype/run_deepaccident_realcoop_alignment_audit.py
prototype/run_deepaccident_realcoop_pilot.py
prototype/run_deepaccident_realmultisource_pilot.py
```

Correction:

```text
source label/lidar
-> source lidar_to_ego
-> source ego_to_world
-> inverse(target ego_to_world)
-> inverse(target lidar_to_ego)
-> target label/lidar

Then remove source-label copies of the target ego vehicle when the aligned
object falls within 2.5 m of the ego origin.
```

Key alignment output:

```text
results/realcoop_alignment_full_val_self_filtered/summary.json
frames = 1753, coverage = 100%
ID residual median = 1.53e-5 m
greedy residual median = 1.51e-5 m
avg real other ego-invisible objects = 12.21/frame
```

Real single-sender update:

```text
results/realcoop_pilot_20x20_fast_self_filtered/summary.csv
results/realcoop_pilot_v1_20x20_self_filtered/summary.csv
results/realcoop_pilot_full_val_fast_self_filtered/summary.csv

20x20 V1:
  EgoOnly 1.850%, CleanCoop oracle 0.425%
  RealOtherRaw 1.725%, RealOtherTrustCalib 1.725%
  RealOtherObjectGuard 1.850%

full-fast:
  EgoOnly 1.888%, CleanCoop oracle 0.291%
  RealOtherRaw 1.580%, RealOtherTrustCalib 1.580%
  RealOtherObjectGuard 1.888%
```

Real multi-source update:

```text
results/realmultisource_20x20_fast_min2_self_filtered_diag/summary.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/frame_diagnostics.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/cluster_records.csv

20x20 min2:
  EgoOnly 1.850%
  CleanCoop oracle 0.425%
  RealPrimaryRaw 1.725%
  RealPrimaryTrustCalib 1.725%
  RealMultiEvidenceGuard 1.625%

missing recovery:
  precision 98.75%, recall 9.12%
  TP/FP = 474/6
  better than EgoOnly on 12/400 frames
  worse than EgoOnly on 6/400 frames
```

Claim update:

```text
The earlier real multi-source result with WPC 2.875% is invalid as a paper
result because it included target-ego duplicate labels as cooperative objects.
It remains useful only as an audit trail showing why real-source input modeling
must be explicitly checked.
```

Paper-ready tables updated:

```text
paper_ready/A_CLASS_EXPERIMENT_FLOW_AND_STATUS_2026-05-30.md
paper_ready/tables/realcoop_results.md
paper_ready/tables/realmultisource_results.md
paper_ready/realcoop_alignment_summary_self_filtered.json
```

Statistical/runtime packaging:

```text
paper_ready/tables/statistical_intervals.md
paper_ready/tables/paired_bootstrap_real_multisource.md
paper_ready/tables/runtime_results.md

synthetic full-val final:
  clean/drop/fake_front/noise+fake_front all 51/17530 = 0.291%
  Wilson 95% CI [0.221%, 0.382%]

runtime probes:
  synthetic final 20x20, two modes: 41.46 s / 800 mode-frames
  real single 20x20: 6.87 s / 400 frames
  real multi-source 20x20: 11.19 s / 400 frames

real multi-source scenario bootstrap:
  RealMultiEvidenceGuard - EgoOnly = -0.225 pp
  95% CI [-1.075 pp, 0.450 pp]
  directionally positive but not a strong real-data superiority claim
```

## 2026-05-30 Path-Risk Real Multi-Source Update

Implemented conservative path-risk-aware one-source missing-object admission:

```text
prototype/run_deepaccident_multipeer_pilot.py
prototype/run_deepaccident_realmultisource_pilot.py

--enable-missing-path-risk-single-support
--missing-path-risk-box-margin-thr 0.0
```

Policy:

```text
strict default: recover missing object if supported by >=2 real sources
path-risk exception: recover a one-source candidate only if its oriented-box
path collision margin is <= 0.0
```

Key outputs:

```text
results/realmultisource_20x20_pathrisk_min2_thr0/
results/realmultisource_20x20_pathrisk_min2_thr5/
results/realmultisource_20x20_pathrisk_temporal_min2/
results/realmultisource_v1_20x20_pathrisk_min2_thr0/
results/realmultisource_full_fast_pathrisk_min2_thr0/
results/realmultisource_full_fast_min2_baseline/
paper_ready/tables/realmultisource_pathrisk_results.md
```

Main finding:

```text
20x20 baseline min2:
  RealMultiEvidenceGuard WPC = 1.625%
  precision = 98.75%, recall = 9.12%

20x20 unconstrained min1:
  RealMultiEvidenceGuard WPC = 1.550%
  precision = 80.50%, recall = 25.25%

20x20 pathrisk-thr0:
  RealMultiEvidenceGuard WPC = 1.550%
  precision = 98.76%, recall = 9.17%
  path-risk single-source TP/FP = 3/0

full-fast pathrisk-thr0:
  RealPrimaryTrustCalib WPC = 1.580%
  RealMultiEvidenceGuard WPC = 1.563%
```

Interpretation:

```text
Path-risk-aware admission keeps the safety-relevant benefit of single-source
recovery without admitting the many far-away single-source false positives.
The temporal variant recovered extra true objects but did not further reduce
WPC, so temporal evidence needs stronger motion consistency before becoming the
recommended real-source policy.
```

## 2026-06-01 Baseline, Bootstrap, And Collusion Update

Added frame-level diagnostics to the synthetic multi-peer script:

```text
prototype/run_deepaccident_multipeer_pilot.py
--write-frame-diagnostics
```

Reran 20x20 V1 baseline/ablation diagnostics:

```text
results/baseline_final_v1_20x20_diag/
results/baseline_no_box_v1_20x20_diag/
results/baseline_no_missing_v1_20x20_diag/
results/baseline_no_smoothing_v1_20x20_diag/
results/baseline_no_noisy_recovery_v1_20x20_diag/
```

Paper tables:

```text
paper_ready/tables/baseline_comparison_20x20_v1.md
paper_ready/tables/scenario_bootstrap_synthetic_20x20_v1.md
```

Key baseline results:

```text
drop:
  Raw fusion / TrustCalib / w/o MissingRecovery = 1.300%
  Final = 0.425%

fake_front:
  Raw fusion / TrustCalib = 2.175%
  Final = 0.425%

noise+fake_front:
  Raw fusion = 1.925%
  TrustCalib = 2.925%
  w/o BoxGuard = 2.425%
  w/o MissingRecovery = 0.700%
  w/o smoothing = 0.450%
  Final = 0.425%
```

Scenario-bootstrap highlights:

```text
noise+fake_front:
  Final - TrustCalib = -2.500 pp, 95% CI [-4.025, -1.250]
  Final - w/o BoxGuard = -2.000 pp, 95% CI [-3.175, -1.075]

fake_front:
  Final - TrustCalib = -1.750 pp, 95% CI [-3.551, -0.400]
```

Collusion stress under final policy:

```text
results/collusion_final_count_min1_v1_20x20_diag/
results/collusion_final_count_min2_v1_20x20_diag/
results/collusion_final_trustweighted_v1_20x20_diag/
paper_ready/tables/collusion_stress_20x20_v1.md
paper_ready/tables/collusion_bootstrap_20x20_v1.md
```

Key collusion results:

```text
support_modes = clean,fake_front

count-min1:
  fake_front = 1.800%
  noise+fake_front = 2.500%

count-min2:
  fake_front = 0.425%
  noise+fake_front = 0.450%

trust-weighted:
  fake_front = 0.425%
  noise+fake_front = 0.425%
```

Interpretation:

```text
The method is not robust to arbitrary one-source collusion. It becomes robust
when either the evidence-count threshold is raised or the colluding sender has
low trust weight. This is an important A-class claim boundary.
```

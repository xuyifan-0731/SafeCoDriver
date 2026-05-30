# Missing-Recovery Precision/Recall Summary

**Date**: 2026-05-28  
**Script**: `/raid/xuyifan/trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py`

## Added Metrics

`summary.csv` now includes direct missing-recovery quality metrics:

```text
avg_missing_gt
missing_recovery_tp
missing_recovery_fp
missing_recovery_precision
missing_recovery_recall
```

Definitions:

```text
missing GT:
  CleanCoop cooperative-only objects that are absent from the primary calibrated message.

TP:
  recovered object geometrically matches one missing GT object.

FP:
  recovered object does not match any missing GT object, or duplicates an already matched GT object.

precision:
  TP / (TP + FP)

recall:
  TP / missing GT
```

Cluster records for missing-recovery rows now also include:

```text
missing_recovery_eval
matched_reference_id
```

## Support-Quality Sweep

All runs:

```text
mode = drop
max_scenarios = 20
max_frames_per_scenario = 20
disable_v1 = true
enable_missing_recovery = true
```

| Support Modes | wp_coll/wp_total | WPC% | Missing GT / Frame | Recovered / Frame | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean+shift | 17/4000 | 0.425 | 12.06 | 11.66 | 4665 | 0 | 1.000 | 0.967 |
| clean+noise | 36/4000 | 0.900 | 11.93 | 6.92 | 2768 | 0 | 1.000 | 0.580 |
| clean+drop | 46/4000 | 1.150 | 11.94 | 3.51 | 1405 | 0 | 1.000 | 0.294 |
| drop+drop | 56/4000 | 1.400 | 11.96 | 1.07 | 429 | 0 | 1.000 | 0.090 |

Interpretation:

```text
Under GT-derived support messages, recovery precision stays 1.0 because support peers do not hallucinate objects.
The main degradation under weaker support is recall, not precision.
```

## Unsafe Single-Peer Recovery Check

Run:

```text
mode = drop
support_modes = clean,fake_front
missing_min_peer_support = 1
missing_min_trust_support = 0.0
```

Result:

| Setting | wp_coll/wp_total | WPC% | Missing GT / Frame | Recovered / Frame | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean+fake_front, min1 | 87/4000 | 2.175 | 12.06 | 12.76 | 4704 | 400 | 0.922 | 0.975 |

Cluster audit:

```text
missing_recovery_eval:
  tp = 4704
  fp = 400

fake-like recovered records:
  400
```

Interpretation:

```text
Single-peer missing recovery can recover fake objects.
The default min_peer_support=2 is therefore a safety requirement, not just a tuning choice.
```

## Outputs

```text
/raid/xuyifan/trusted_coop_perception/results/recovery_pr_drop_clean_shift_20x20/
/raid/xuyifan/trusted_coop_perception/results/recovery_pr_drop_clean_noise_20x20/
/raid/xuyifan/trusted_coop_perception/results/recovery_pr_drop_clean_drop_20x20/
/raid/xuyifan/trusted_coop_perception/results/recovery_pr_drop_drop_drop_20x20/
/raid/xuyifan/trusted_coop_perception/results/recovery_pr_drop_clean_fake_min1_20x20/
```

## Paper-Facing Claim

Safe claim:

```text
Missing-object recovery achieves high precision when at least two calibrated reliable peers support the object, but its recall depends on support quality. Single-peer recovery is unsafe under hallucinating or malicious support and should not be used as an adversarial default.
```

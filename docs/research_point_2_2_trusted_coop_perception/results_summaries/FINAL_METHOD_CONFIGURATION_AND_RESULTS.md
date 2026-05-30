# Final Method Configuration And Results

**Date**: 2026-05-28  
**Script**: `/raid/xuyifan/trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py`  
**Environment**: `Android-Lab`

**Update**: This file is superseded for the noise boundary by
`/raid/xuyifan/trusted_coop_perception/results/CONSENSUS_SMOOTHING_FINAL_UPDATE_2026-05-28.md`.
PeerConsensusSmoothing plus evidence-gated recovery from downweighted/quarantined
primary messages now recovers clean-level WPC for `noise+fake_front` in 20x20,
full-frame, and accident-window checks.

## Final Prototype Configuration

Paper-facing method name:

```text
TrustCalib + MultiPeerObjectGuard + MissingRecovery + BoxGuard + TimeCalib
```

Recommended synthetic-pilot command shape:

```bash
python prototype/run_deepaccident_multipeer_pilot.py \
  --support-modes clean,shift \
  --enable-time-calib \
  --enable-missing-recovery \
  --missing-recovery-primary-actions accept,correct,time_correct \
  --enable-box-margin-guard
```

Module responsibilities:

| Module | Trigger | Action | Handles | Main Caveat |
|---|---|---|---|---|
| TrustCalib | stable spatial offset | translate cooperative message | shift, shift_severe | assumes enough visible anchors |
| TimeCalib | visible anchors fit delayed motion better than spatial correction | propagate message by estimated delay | stale | constant-velocity approximation |
| MultiPeerObjectGuard | unsupported high-impact coop-only object | object quarantine | fake_front | evidence gate must be robust to collusion |
| MissingRecovery | primary message is trusted/corrected/time-corrected but incomplete | recover peer-supported missing objects | drop, shift+drop, stale+drop | depends on reliable support peers |
| BoxGuard | unsupported object intrudes into oriented path occupancy | object quarantine | residual fake leakage | supported colluding fake can bypass count-only support |

## TimeCalib Result

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/time_calib_stale_core_20x20_fast/summary.csv
```

Subset:

```text
20 validation scenarios x first 20 frames, V1 disabled
```

| Mode | WPC% | Warn% | Time Correct Rate | Avg Delay | Key Result |
|---|---:|---:|---:|---:|---|
| clean | 0.425 | 46.50 | 0.0% | 0.000 | no false trigger |
| shift | 0.425 | 46.50 | 0.0% | 0.000 | spatial correction retained |
| stale | 0.425 | 46.50 | 99.0% | 0.990 | stale restored to clean level |
| drop | 0.425 | 46.50 | 0.0% | 0.000 | missing recovery retained |
| fake_front | 0.425 | 46.50 | 0.0% | 0.000 | fake removal 100.0% |

## Compound-Anomaly Result

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/compound_anomaly_corrected_recovery_20x20_fast/summary.csv
```

Subset:

```text
20 validation scenarios x first 20 frames, V1 disabled
```

| Mode | WPC% | Warn% | Missing Recovered / Frame | Fake Removal | Interpretation |
|---|---:|---:|---:|---:|---|
| shift+drop | 0.425 | 46.50 | 11.66 | 0.0% | corrected then recovered |
| stale+drop | 0.425 | 46.50 | 11.67 | 0.0% | time-corrected then recovered |
| shift+fake_front | 0.525 | 47.50 | 0.00 | 93.7% | mostly controlled, small residual |
| noise+fake_front | 0.725 | 45.50 | 0.12 | 99.4% | still above clean; noise remains open |

Important setting:

```text
--missing-recovery-primary-actions accept,correct,time_correct
```

This is needed for combined shift/stale + drop, because after spatial/time correction the message is usable but incomplete.

## V1-Enabled 20x20 Check

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/final_geometric_policy_v1_20x20/summary.csv
```

| Mode | WPC% | Warn% | Avg P(collision) | Key Result |
|---|---:|---:|---:|---|
| clean | 0.425 | 56.50 | 0.228 | baseline |
| shift | 0.425 | 57.50 | 0.233 | corrected |
| stale | 0.425 | 56.50 | 0.228 | time-corrected |
| drop | 0.425 | 57.00 | 0.234 | recovered |
| fake_front | 0.425 | 56.50 | 0.228 | fake removal 100.0% |
| shift+drop | 0.425 | 57.75 | 0.237 | corrected + recovered |
| stale+drop | 0.425 | 57.00 | 0.234 | time-corrected + recovered |
| noise+fake_front | 0.725 | 57.00 | 0.237 | partially controlled, not solved |

## Full-Frame Fast Check

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/final_geometric_policy_full_val_fast/summary.csv
```

Subset:

```text
22 validation scenarios, all frames
frames = 1753
wp_total = 17530
V1 disabled
```

| Mode | wp_coll/wp_total | WPC% | Warn% | Key Result |
|---|---:|---:|---:|---|
| clean | 51/17530 | 0.291 | 43.87 | oracle upper bound |
| stale | 52/17530 | 0.297 | 43.87 | near clean after TimeCalib |
| drop | 51/17530 | 0.291 | 43.87 | clean-level after recovery |
| fake_front | 51/17530 | 0.291 | 43.87 | fake removal 99.8% |
| shift+drop | 51/17530 | 0.291 | 43.87 | clean-level |
| stale+drop | 51/17530 | 0.291 | 43.87 | clean-level |
| noise+fake_front | 125/17530 | 0.713 | 41.47 | open issue |

## Remaining Boundary

Noise remains the main unresolved anomaly:

```text
noise+fake_front:
  20x20 V1 WPC = 0.725%
  full-frame fast WPC = 0.713%
```

The current method can remove the fake object, but noisy real-object positions still degrade downstream waypoint safety. This should be handled with covariance-aware fusion, robust smoothing, or trust penalties rather than missing recovery.

## Paper-Facing Claim

Safe claim:

```text
In synthetic DeepAccident object-list perturbation pilots, the final prototype
recovers clean-level WPC for spatial shift, stale delay, dropout, fake-front
injection, and shift/stale+drop compound anomalies when reliable GT-derived
support peers are available. Noise and colluding support remain explicit
boundary cases.
```

Avoid:

```text
full real multi-vehicle validation
robust to all attacks
solves noise/fault/collusion in general
```

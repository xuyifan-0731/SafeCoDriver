# Corrected Results And Support Sweeps

**Date**: 2026-05-28  
**Reason**: fix misleading `50x20` naming and add support-quality/collusion stress checks.

## Script Update

`run_deepaccident_multipeer_pilot.py` now writes explicit raw-count and sample-size fields to `summary.csv`:

```text
wp_coll
wp_total
actual_scenarios
available_val_scenarios
requested_max_scenarios
max_frames_per_scenario
enable_missing_recovery
enable_box_margin_guard
disable_v1
```

It also writes `val_scenario_idx_used` into `metadata.json` and prints a warning when `--max-scenarios` exceeds the available validation scenario list.

## Corrected 22x20 Default

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/deepaccident_final_default_22x20_fast/summary.csv
```

Actual sample:

```text
actual_scenarios = 22
frames = 440
wp_total = 4400
disable_v1 = true
support_modes = clean,shift
```

| Mode | MultiPeerObjectGuard wp_coll/wp_total | WPC% | Warn% | Key Result |
|---|---:|---:|---:|---|
| clean | 17/4400 | 0.386 | 48.18 | same as CleanCoop |
| shift | 17/4400 | 0.386 | 48.18 | corrected to clean level |
| shift_severe | 17/4400 | 0.386 | 48.18 | corrected to clean level |
| noise | 25/4400 | 0.568 | 47.27 | still open |
| drop | 52/4400 | 1.182 | 43.41 | still open without missing recovery |
| stale | 44/4400 | 1.000 | 45.68 | still open |
| fake_front | 19/4400 | 0.432 | 48.18 | 93.6% fake removal |

## Corrected 22x20 Missing Recovery

Output:

```text
/raid/xuyifan/trusted_coop_perception/results/deepaccident_missing_recovery_22x20_fast/summary.csv
```

| Mode | MultiPeerObjectGuard wp_coll/wp_total | WPC% | Warn% | Missing Recovered / Frame | Key Result |
|---|---:|---:|---:|---:|---|
| clean | 17/4400 | 0.386 | 48.18 | 0.00 | no degradation |
| drop | 17/4400 | 0.386 | 48.18 | 11.88 | recovers clean-level WPC with reliable support |
| fake_front | 19/4400 | 0.432 | 48.18 | 0.00 | fake guard unchanged |

## Drop Support-Quality Sweep

All runs:

```text
mode = drop
max_scenarios = 20
max_frames_per_scenario = 20
disable_v1 = true
enable_missing_recovery = true
```

| Support Modes | wp_coll/wp_total | WPC% | Missing Recovered / Frame | Avg Real Support | Interpretation |
|---|---:|---:|---:|---:|---|
| clean+shift | 17/4000 | 0.425 | 11.66 | 2.0 | near-oracle support; strong recovery |
| clean+noise | 36/4000 | 0.900 | 6.92 | 1.6 | partial recovery under noisy support |
| clean+drop | 46/4000 | 1.150 | 3.51 | 1.3 | weak recovery |
| drop+drop | 56/4000 | 1.400 | 1.07 | 0.7 | recovery largely fails |

Conclusion:

```text
missing recovery is valid as an availability mechanism when reliable peers exist;
it is not robust to global peer dropout by itself.
```

Precision/recall follow-up:

```text
results/RECOVERY_PRECISION_RECALL_SUMMARY.md
```

The support-quality sweep shows precision stays 1.000 under GT-derived non-hallucinating support, while recall drops from 0.967 under clean+shift support to 0.090 under drop+drop support.

Unsafe single-peer recovery check:

```text
support_modes=clean,fake_front, missing_min_peer_support=1:
  WPC 2.175%
  TP 4704, FP 400
  precision 0.922, recall 0.975
```

This confirms that the default `missing_min_peer_support=2` is a safety requirement.

## Fake-Front Collusion Sweep

All runs:

```text
mode = fake_front
max_scenarios = 20
max_frames_per_scenario = 20
disable_v1 = true
enable_box_margin_guard = true
```

| Support Modes | Evidence Gate | wp_coll/wp_total | WPC% | Fake Removal | Avg Fake Support | Interpretation |
|---|---|---:|---:|---:|---:|---|
| clean+shift | min_peer_support=1 | 17/4000 | 0.425 | 100.0% | 0.0 | no fake support; succeeds |
| clean+fake_front | min_peer_support=1 | 87/4000 | 2.175 | 0.0% | 1.0 | one colluder can defeat support gate |
| clean+fake_front | min_peer_support=2 | 17/4000 | 0.425 | 100.0% | 1.0 | requires two supports; succeeds |
| fake_front+fake_front | min_peer_support=2 | 87/4000 | 2.175 | 0.0% | 2.0 | full collusion defeats count gate |
| clean+fake_front | support_trusts=1.0,0.2; min_trust_support=1.0 | 17/4000 | 0.425 | 100.0% | 1.0 | trust-weighted gate succeeds |

Conclusion:

```text
BoxGuard only removes unsupported path-intruding objects.
Adversarial settings require either stricter peer-count gates or trust-weighted evidence.
```

## Deprecated / Historical Naming

These previous directories are retained for traceability but should not be cited as `50x20`:

```text
/raid/xuyifan/trusted_coop_perception/results/deepaccident_final_default_50x20_fast
/raid/xuyifan/trusted_coop_perception/results/deepaccident_missing_recovery_50x20_fast
```

They contain 440 evaluated frames, not 1000 frames.

## Paper-Facing Use

Use these phrases:

```text
22-scenario validation-split subset, first 20 frames per scenario.
Synthetic peer-evidence ablation using GT-derived support messages.
CleanCoop is an oracle cooperative-perception upper bound.
```

Avoid:

```text
50 validation scenarios.
independent multi-vehicle validation.
full DeepAccident validation.
```

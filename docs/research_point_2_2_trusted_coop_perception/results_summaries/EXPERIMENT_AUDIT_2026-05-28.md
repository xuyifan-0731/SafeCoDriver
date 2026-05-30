# Experiment Audit

**Date**: 2026-05-28  
**Scope**: `/raid/xuyifan/trusted_coop_perception` DeepAccident trusted cooperative perception pilots  
**Conclusion**: the current results are useful prototype evidence, but they should not yet be presented as full, realistic, or final validation.

## Executive Findings

1. The main positive claims are directionally supported under the synthetic setup:

```text
spatial shift -> TrustCalib correction works
fake_front -> unsupported high-impact object quarantine works
drop with clean peers -> missing-object recovery can restore WPC
```

2. Several settings are too favorable and must be labeled as synthetic/upper-bound:

```text
support peers are generated from the same GT object list
CleanCoop is a GT cooperative upper bound
support_modes=clean,shift provides near-oracle evidence after calibration
```

3. Some result names/descriptions are misleading:

```text
"50x20" runs used only 22 validation scenarios, not 50, because checkpoint val_scenario_idx has length 22.
Actual frame count = 22 x 20 = 440 frames.
```

4. The current evaluation is not yet a full validation:

```text
most runs use first 20 frames only
most fast runs disable V1
only one random seed is used
attack/fault modes are mostly single-anomaly synthetic modes
```

## Hard Issues To Fix Before Paper Tables

### 1. "50x20" Is Actually 22x20

Code path:

```text
val_idx = list(ckpt.get("val_scenario_idx", []))
if args.max_scenarios > 0:
    val_idx = val_idx[: args.max_scenarios]
```

Observed:

```text
len(val_scenario_idx) = 22
deepaccident_final_default_50x20_fast: frames = 440
deepaccident_missing_recovery_50x20_fast: frames = 440
```

Impact:

```text
The metrics are not wrong, but "50x20" and "first 50 validation scenarios" are wrong descriptions.
```

Required fix:

```text
rename/report as 22x20 val-split subset, or switch to an explicit scenario list with 50 scenarios.
```

### 2. Multi-Peer Evidence Is Synthetic And Not Independent

Current setup:

```text
loader = DeepAccidentLoader(split="all", include_invisible=True, include_coop=False)
primary/support messages are produced by inject_anomaly(frame.perception, ...)
support_modes=clean,shift
```

This means both support peers come from the same ego GT perception object list. The shifted support peer becomes nearly identical to clean after calibration.

Impact:

```text
support_count=2 should not be described as two independent vehicles.
It is synthetic evidence from duplicated GT object lists under controlled transforms.
```

Required fix:

```text
Use real other_vehicle labels where available, or clearly label these as synthetic peer-evidence ablations.
Add experiments with support peer faults: drop/drop, noise/drop, stale/clean, colluding fake peers.
```

### 3. Missing Recovery Depends On Clean/Oracle Support

Positive result:

```text
support_modes=clean,shift:
  drop WPC 1.30% -> 0.43%, missing recovery 11.66/frame
```

Audit stress:

```text
support_modes=drop,drop:
  PrimaryTrustCalib WPC 1.38%
  MultiPeerObjectGuard WPC 1.40%
  missing recovery 1.07/frame
```

Impact:

```text
The current missing recovery result demonstrates an availability mechanism when reliable peers exist.
It does not prove robustness when the whole peer set suffers dropout.
```

Required fix:

```text
Report clean/shift support as an upper-bound setting.
Add a support-quality sweep: clean+shift, clean+drop, drop+drop, clean+noise, drop+noise.
```

### 4. BoxGuard Is Not Collusion-Robust With min_peer_support=1

Positive result:

```text
support_modes=clean,shift:
  fake removal 100.0%, WPC 0.43%
```

Audit stress:

```text
support_modes=clean,fake_front, min_peer_support=1:
  MultiPeerObjectGuard WPC 2.17%
  fake removal 0.0%

support_modes=clean,fake_front, min_peer_support=2:
  MultiPeerObjectGuard WPC 0.43%
  fake removal 100.0%
```

Impact:

```text
BoxGuard removes unsupported path-intruding objects.
If one colluding peer is enough to mark the fake as supported, BoxGuard will keep it.
```

Required fix:

```text
For adversarial settings, min_peer_support=2 or trust-weighted support threshold must be part of the default.
Report collusion stress explicitly.
```

## Methodological Caveats

### 5. CleanCoop Is An Upper Bound, Not A Real Baseline

Current CleanCoop:

```text
merge_ego_visible_with_coop_invisible(frame.perception, frame.perception)
```

Because `frame.perception` includes invisible GT agents, CleanCoop is effectively an oracle cooperative perception upper bound.

Use it as:

```text
upper-bound reference
```

Do not describe it as:

```text
real cooperative perception baseline
```

### 6. Evaluation Uses First Frames, Not Full Scenarios

Most pilot runs use:

```text
--max-frames-per-scenario 20
```

This evaluates the first 20 frames of each selected scenario, not necessarily the accident-critical window.

Impact:

```text
WPC values are useful for quick comparison, but not full scenario safety performance.
```

Required fix:

```text
Run full validation frames or accident-window evaluation around collision_frame.
```

### 7. WPC Percentages Are Small-Count Metrics

Example:

```text
400 frames x 10 waypoints = 4000 waypoint checks
0.43% WPC ~= 17 waypoint collisions
```

Impact:

```text
Small absolute count changes can move the percentage.
```

Required fix:

```text
Report raw wp_coll/wp_total counts and bootstrap or scenario-level confidence intervals.
```

### 8. Warn Rate Is Not False Alarm Rate

Current warn flag:

```text
warned = n_collisions_detected > 0 or modification_rate > 0 or n_geometric_threats > 0
```

Impact:

```text
warn_rate is a frame-level intervention/warning proxy across the evaluated subset.
It is not a standard false alarm rate unless split by normal scenarios and warning semantics.
```

Required fix:

```text
Keep warn_rate wording.
Use existing unified metrics for CDR/FA or add scenario-level accident/normal split.
```

### 9. Fast Runs With V1 Disabled Should Not Be Mixed With V1 Runs

Fast runs use:

```text
--disable-v1
avg_p_coll = 0.0
```

Impact:

```text
They validate geometric correction and object guard behavior, not learned-detector behavior.
```

Required fix:

```text
Separate "fast geometric-only ablation" from "V1-enabled method result".
```

### 10. Attack/Fault Modes Are Too Simple

Current synthetic modes:

```text
shift: global translation
noise: i.i.d. Gaussian position noise
drop: invisible-agent Bernoulli dropout
stale: x -= vx * delay, y -= vy * delay
fake_front: one stationary fake object on ego path
```

Impact:

```text
These are good unit tests, but they do not cover mixed anomalies, adaptive attacks, fake side/rear objects, fake moving objects, partial collusion, calibration drift, or type/size spoofing.
```

Required fix:

```text
Add compound modes: shift+drop, shift+fake, stale+drop, noise+fake.
Add fake variants: lateral, moving, size-scaled, multiple fakes.
```

## Code/Metric Issues

### 11. Recovered-Object False Positives Are Not Counted Directly

`real_removed` compares primary objects before/after guard:

```text
count_real_removed(primary_trust, guarded)
```

Recovered objects are not in `primary_trust`, so false recovered objects are not directly counted.

Impact:

```text
If support peers hallucinate objects, current summary may not expose recovery false positives except indirectly through WPC/warn/modification.
```

Required fix:

```text
Add recovered object precision/recall against clean GT object list using geometry matching.
```

### 12. BoxGuard Actions Are Not Separately Labeled In Summary

Current `guarded_report.action` remains:

```text
object_quarantine
```

for both original ObjectGuard and BoxGuard.

Impact:

```text
summary.csv cannot distinguish geom/mod quarantine from box-margin quarantine.
```

Required fix:

```text
Add separate counters: box_margin_quarantine, impact_quarantine.
```

### 13. Thresholds Are Tuned On The Same Pilot Subsets

Examples:

```text
peer_match_dist=2.5
min_peer_support=1 or 2
missing_min_peer_support=2
box_margin_guard_thr=0.0
geom_delta_thr=3.0
mod_delta_thr=0.25
```

Impact:

```text
Good for prototyping, not final evidence.
```

Required fix:

```text
Use train/tune/eval split for thresholds, or report threshold sensitivity.
```

## Recommended Reporting Language

Safe phrasing:

```text
On synthetic DeepAccident object-list perturbation pilots, the proposed availability framework shows that spatially shifted messages can be corrected, unsupported path-intruding fake objects can be quarantined, and missing objects can be recovered when reliable peer evidence is available.
```

Avoid:

```text
The method is proven robust to malicious attacks, dropout, and stale messages.
The multi-peer evidence is validated on real independent vehicles.
The 50x20 result evaluates 50 scenarios.
```

## Priority Fix Plan

1. Rename or regenerate all "50x20" results.
2. Add true scenario count and raw wp_coll/wp_total to every summary table.
3. Add support-quality sweeps:

```text
clean+shift
clean+drop
drop+drop
clean+noise
clean+fake_front
fake_front+fake_front
```

4. Add compound anomaly modes:

```text
shift+drop
shift+fake_front
stale+drop
noise+fake_front
```

5. Add recovered-object precision/recall.
6. Add full-frame or accident-window validation.
7. Run V1-enabled full or cached validation.
8. Keep old results as prototype logs, but do not use them as final paper tables without caveats.

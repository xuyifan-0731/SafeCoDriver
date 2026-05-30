# Method Constraints After Prior-Work Check

**Date**: 2026-05-27

## What The Method Must Not Be

The method must not be presented as:

- a new generic V2X trust management protocol;
- a generic CPM misbehavior reporting system;
- a new pose-error-robust feature fusion backbone;
- a covariance-weighted track fusion method;
- a communication-interruption recovery model.

These are already covered or partially covered by prior work.

## What The Method Should Be

A lightweight safety-facing pre-fusion layer:

```text
cooperative object messages
  -> association
  -> residual offset estimation
  -> correctability scoring
  -> usability/trust scoring
  -> safety-impact evaluation
  -> calibrated PerceptionResult
```

## Required Differentiators

### D1. Explicit Correctability

The method must distinguish:

```text
stable spatial bias -> correct
random noise -> downweight
bogus object / nonphysical jump -> quarantine
missing object -> reduce trust / rely on other sources
```

### D2. Explicit Residual Calibration

At least the first implementation should output:

```text
dx, dy
residual_before
residual_after
inlier_ratio
match_count
correctable_score
```

If rotation is included:

```text
dtheta
offset_covariance
```

### D3. Downstream Safety Impact

The trust update should include SafeCoDriver deltas:

```text
safety_impact =
  w1 * |collision_prob_with - collision_prob_without|
+ w2 * normalized_delta(min_ttc)
+ w3 * |n_geometric_threats_with - n_geometric_threats_without|
+ w4 * |modification_rate_with - modification_rate_without|
+ w5 * |target_speed_factor_with - target_speed_factor_without|
```

High-impact inconsistent messages should be penalized more than low-impact noisy messages.

### D4. Calibration Beats Discard

The first main experiment must show:

```text
TrustCalib > Hard-Filter
```

under stable injected spatial shifts. Without this, the paper's core argument is weak.

## Minimal Viable Prototype

Only implement enough for a decisive pilot:

1. Use DeepAccident ego perception as reference.
2. Treat `coop_perception` as the cooperative message.
3. Inject known shifts/noise/fake objects.
4. Match objects by class + distance + size.
5. Estimate translation offset with robust inlier filtering.
6. Score correctability.
7. Apply accept/correct/downweight/quarantine.
8. Run existing `HybridSafetyConstraint`.

## Pilot Stop/Go

Proceed to full trust/evidence design only if:

```text
medium shift dx=2m, dy=1m:
  residual_after <= 40% of residual_before
  WPC% improves over Raw-Coop
  TrustCalib retains better useful-info coverage than Hard-Filter
```

If these fail, the topic should pivot to diagnostic evaluation rather than a correction method.

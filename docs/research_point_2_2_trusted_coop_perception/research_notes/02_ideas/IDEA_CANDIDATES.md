# Idea Candidates

## Idea 1: TrustCalib Layer for Object-Level Cooperative Perception

**One-line thesis**: Estimate residual source offsets and message usability before fusion, then correct/downweight/quarantine cooperative messages based on both evidence consistency and downstream safety impact.

### Mechanism

```text
ego perception + coop messages
  -> object association
  -> robust SE(2) residual offset estimation
  -> correctability score
  -> message usability score
  -> vehicle trust update
  -> calibrated PerceptionResult
```

### Why It Is Promising

- Directly matches the user's research point.
- Can be implemented on current SafeCoDriver without retraining the V1 detector.
- Evaluates with existing DeepAccident WPC/FA and SUMO CollRate metrics.
- Strong story: "do not discard useful but shifted information."

### Risks

- Object-level GT perception may make the task look too engineered if not framed carefully.
- Need distinguish from ordinary pose-error robustness papers.
- Need enough multi-source evidence in DeepAccident/SUMO to support trust updates.

### Fast Pilot

Inject known shifts into cooperative objects:

```text
dx, dy in {0.5, 1.0, 2.0, 4.0} m
dtheta in {0, 3, 6} deg
```

Compare:

```text
Raw-Coop vs Ego-Only vs Hard-Filter vs Oracle-Calib vs TrustCalib
```

Metrics:

```text
Offset-MAE, residual-before/after, WPC%, FA(f), Det(s), Mod%
```

## Idea 2: Safety-Impact-Aware Trust Update

**One-line thesis**: Trust penalties should scale with how much a message changes safety decisions, not just how inconsistent it is geometrically.

### Mechanism

For each message, compute Hybrid stats with and without it:

```text
impact = |collision_prob_with - collision_prob_without|
       + a * |min_ttc_with - min_ttc_without|
       + b * |n_geometric_threats_with - n_geometric_threats_without|
       + c * |modification_rate_with - modification_rate_without|
```

Use `impact` as a multiplier in the trust update. A low-impact noisy message should not destroy long-term trust; a high-impact forged obstacle should be penalized aggressively.

### Why It Is Promising

- Bridges research point 2.2 with SafeCoDriver's existing safety outputs.
- Strong differentiator from perception-only trust scores.

### Risks

- More expensive because it may run safety evaluation multiple times per frame.
- Need avoid circularity: a message can be high-impact because it is genuinely important.

## Idea 3: Minimal Evidence Exchange for Cooperative Trust Consensus

**One-line thesis**: Vehicles do not exchange raw sensor data for trust reasoning; they exchange residual summaries, offset estimates, trust posteriors, and evidence hashes.

### Mechanism

```text
EvidenceMessage = {
  issuer_id,
  target_id,
  offset_mean/cov,
  residual_mean/p95,
  inlier_ratio,
  match_count,
  trust_alpha/beta,
  recommended_action,
  evidence_hash
}
```

Fuse peer evidence using:

```text
weight = issuer_trust * evidence_quality * freshness * viewpoint_diversity
```

### Why It Is Promising

- Fits the user's "最小充分信息：偏移值、可信度和证据链".
- Explains decisions, not just outputs scores.

### Risks

- Harder to evaluate with existing two-view DeepAccident data.
- SUMO may be better for multi-vehicle synthetic evidence.

## Recommended Initial Direction

Start with Idea 1 and add Idea 2 as the main differentiating feature. Keep Idea 3 as an extension once the object-level correction pilot works.

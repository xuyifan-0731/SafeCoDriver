# Object Evidence Chain Design

**Topic**: 空间失配场景下考虑异常信息的安全可信协同感知优化  
**Current prototype**: `/raid/xuyifan/trusted_coop_perception/prototype/`  
**Latest pilots**:

```text
/raid/xuyifan/trusted_coop_perception/results/OBJECT_GUARD_SUMMARY.md
/raid/xuyifan/trusted_coop_perception/results/MULTI_PEER_OBJECT_GUARD_SUMMARY.md
/raid/xuyifan/trusted_coop_perception/results/CLUSTER_EVIDENCE_SUMMARY.md
```

## Core Idea

Do not make cooperative perception a binary normal/abnormal decision. Use a layered information-availability score:

```text
availability(vehicle, message, object)
  = vehicle trust
  + message correctability
  + object evidence support
  + downstream safety impact
```

The key design shift after the pilot is:

```text
message-level calibration decides whether the source geometry is correctable;
object-level evidence decides whether each cooperative-only object is usable;
safety impact decides whether unsupported objects must be quarantined.
```

## Pipeline

### 1. Message-Level Spatial Correctability

Input:

```text
ego perception E_t
cooperative message M_i,t
```

Steps:

1. Associate ego-visible objects with cooperative visible objects without using object IDs.
2. Estimate translation offset:

```text
Delta_i,t = median/mean_inlier(E_visible - M_visible)
```

3. Compute residual before and after correction.
4. Output action:

| Action | Condition | Fusion behavior |
|---|---|---|
| `accept` | clean residual, enough matches | keep cooperative blind-spot objects |
| `correct` | high correctability | apply offset to cooperative objects |
| `downweight` | uncertain but partially useful | apply offset with lower confidence |
| `quarantine` | low correctability | discard message-level cooperative objects |

Prototype file:

```text
prototype/trust_calib.py
```

### 2. Object-Level Evidence Chain

For each cooperative-only object `o_j` in an accepted/corrected message, build an evidence record:

```text
object_evidence(o_j) = {
  sender_id,
  corrected_position,
  message_offset,
  temporal_track_support,
  peer_support_count,
  source_confidence,
  evidence_chain
}
```

Evidence support should come from at least one of:

1. self temporal continuity from accepted historical tracks;
2. peer corroboration from another sender after spatial correction;
3. sender-side evidence chain such as raw detection metadata;
4. ego-visible association.

Latest finding:

```text
single-vehicle temporal continuity alone is insufficient;
peer corroboration is the stronger route.
```

### 3. Safety-Impact Attribution

For unsupported cooperative-only objects, estimate marginal downstream impact:

```text
impact(o_j) = SafetyMetric(E_t + o_j) - SafetyMetric(E_t)
```

Current pilot uses geometric waypoint safety:

```text
geom_delta = n_geometric_threats(E_t + o_j) - n_geometric_threats(E_t)
mod_delta  = waypoint_modification_rate(E_t + o_j) - waypoint_modification_rate(E_t)
```

Decision:

| Evidence | Safety impact | Action |
|---|---|---|
| supported | any | keep/correct/downweight according to message-level action |
| unsupported | low | keep temporarily or downweight |
| unsupported | high | object-level quarantine |

This avoids whole-message fallback and preserves useful cooperative objects.

Prototype file:

```text
prototype/run_deepaccident_mixed_pilot.py
```

## Current Experimental Evidence

On 20 validation scenarios x 20 frames with V1 disabled:

| Setting | Result |
|---|---|
| stable shift | ObjectGuard WPC 0.43%, same as CleanCoop; corrects 99% |
| severe shift | ObjectGuard WPC 0.43%, same as CleanCoop; corrects 97% |
| fake_front attack | Raw/TrustCalib WPC 2.17%; ObjectGuard WPC 0.40% |
| peer evidence | reproduces fake_front mitigation without checking fake ID/source |
| temporal-only evidence | negative result; over-quarantines clean high-impact objects |
| explicit multi-peer evidence | clean/shift/severe shift WPC 0.43%, fake_front WPC 0.40%, real false removal 0.000/frame |
| colluding peer stress | 1-of-2 support fails under one fake support peer; 2-of-2 support recovers fake removal |
| trust-weighted cluster evidence | low-trust colluding support has fake trust support 0.2; threshold 1.0 recovers fake removal |

## Next Implementation Step

The explicit multi-peer and cluster evidence simulation is now implemented in:

```text
prototype/run_deepaccident_multipeer_pilot.py
```

Next, replace pairwise matching with cluster-level evidence aggregation:
Cluster-level metrics already include:

```text
support_count
trust_weighted_support
position_cov_trace
offset_spread
```

Next, export per-frame cluster records:

1. Generate `K` cooperative messages from the same frame.
2. Apply independent anomaly modes to each peer:

```text
peer_1: shift/noise/stale/drop/fake_front
peer_2: clean or independently shifted/noisy
peer_3: optional attacker/faulty peer
```

3. Calibrate each peer message into ego coordinates.
4. Cluster cooperative-only objects across peers.
5. For each cluster, write:

```text
peer_support_count
supporting_sender_ids
offset_consistency
position_covariance
source_trust_weighted_vote
safety_impact
final_action
```

6. Run safety-impact quarantine only for high-impact clusters with weak support.

Expected new metrics:

| Metric | Purpose |
|---|---|
| fake removal recall | how many injected fake objects were removed |
| clean object false removal | how many real cooperative objects were wrongly removed |
| WPC | downstream waypoint collision risk |
| warn/FA with V1 enabled | detection false alarm behavior |
| average corrected offset | spatial calibration quality |

## Paper-Level Claim To Target

The defensible claim is:

```text
The proposed method estimates cooperative information availability at message
and object granularity. Unlike whole-message filtering, it can correct spatial
misalignment, preserve useful blind-spot information, and selectively suppress
unsupported high-impact objects that would otherwise corrupt downstream safety.
```

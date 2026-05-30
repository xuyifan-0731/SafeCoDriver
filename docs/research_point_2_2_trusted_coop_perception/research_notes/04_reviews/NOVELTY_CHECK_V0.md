# Novelty Check V0

**Date**: 2026-05-27  
**Status**: first-pass, not final. Needs deeper title-level verification and reading of closest papers.

## Proposed Method

A pre-fusion trust and calibration layer for cooperative perception. It evaluates long-term source trust, current message usability, and correctability; estimates residual spatial offsets; exchanges minimal evidence; and outputs calibrated `PerceptionResult` objects for SafeCoDriver.

## Core Claims

| Claim | Novelty Risk | Closest Prior Work | Current Assessment |
|---|---|---|---|
| Unified information availability across attacks, faults, misalignment, and noise | Medium | surveys and empirical V2X error studies; Byzantine sensor fusion | The unifying framing is plausible, but vehicular trust-management literature may overlap. |
| Explicit correctability score + residual offset correction before downstream safety | Medium/High | V2X-DGPE, V2X-ViT pose-error robustness | Need emphasize object/message-level calibration and safety output, not feature-level robustness. |
| Safety-impact-aware trust update using planner/safety-module response | High | fault-tolerance testing of MSF perception; end-to-end V2X safety papers | This looks like the strongest differentiator if implemented cleanly. |
| Minimal evidence exchange with offset/trust/evidence chain | Medium/High | VANET trust/reputation systems, V2X misbehavior standardization, Byzantine fusion | This must be positioned carefully: evidence exchange is not novel by itself; object-level offset/evidence plus downstream safety impact is the differentiator. |

## Closest Prior Work Clusters

### 1. Pose-Error-Robust Cooperative Perception

- V2X-ViT handles asynchronous sharing, pose errors, and heterogeneous V2X components.
- V2X-DGPE directly addresses domain gaps and pose errors using historical information and deformable attention.

**Risk**: reviewers may say spatial misalignment has already been studied.  
**Differentiator needed**: this work is not a new fusion backbone; it estimates runtime message-level residual offsets and outputs correction/usability decisions.

### 2. Communication Interruption and Missing Information

- V2X-INCOP recovers missing cooperative information under communication interruption.
- Selective communication chooses useful cooperative agents under bandwidth limits.

**Risk**: availability may be confused with communication availability.  
**Differentiator needed**: availability here includes correctness, trust, and correctability, not only whether a message arrives.

### 3. Fault Injection and System-Level Safety

- FADE-like work tests multi-sensor fusion under camera/LiDAR faults and system-level driving behavior.

**Risk**: system-level fault testing already exists.  
**Differentiator needed**: V2X cooperative messages, residual calibration, and SafeCoDriver downstream metrics.

### 4. Byzantine and Trust-Based Fusion

- Byzantine sensor fusion theory exists.
- VANET trust/reputation literature likely has many schemes for malicious nodes.
- `V2X Misbehavior and Collective Perception Service: Considerations for Standardization` is a new red-flag seed because it is directly about V2X misbehavior and collective perception.

**Risk**: evidence exchange and trust update may be old if presented generically.  
**Differentiator needed**: object-level perception evidence, residual spatial offset, and downstream safety impact.

## Current Novelty Score

```text
Method as originally phrased: 5.0 / 10
Method with safety-impact-aware correctability: 7.0 / 10
```

## Recommended Positioning

Do not frame the paper as:

```text
"We propose a trust mechanism for V2X."
"We solve pose errors in cooperative perception."
```

Frame it as:

```text
"We introduce a safety-facing information availability layer for cooperative perception that explicitly separates source trust, current message usability, and anomaly correctability. The layer preserves useful but spatially biased information through residual calibration, while preventing high-impact abnormal messages from corrupting downstream safety constraints."
```

## Must-Do Follow-Up Searches

1. "VANET trust management misbehavior detection cooperative perception"
2. "secure cooperative perception autonomous driving malicious vehicle"
3. "Byzantine resilient cooperative perception autonomous driving"
4. "V2X misbehavior detection object-level perception"
5. "trust aware sensor fusion autonomous driving"
6. "ETSI collective perception service misbehavior detection"

## Red Flags To Resolve

- Whether a recent paper already has "trust-aware cooperative perception" with object-level correction.
- Whether pose-error papers already estimate and correct residual transforms explicitly.
- Whether evidence exchange is too close to standard vehicular reputation systems.
- Whether DeepAccident two-view data is sufficient to support "multi-vehicle evidence exchange" claims.
- Whether standardization work on collective perception misbehavior already defines enough evidence fields to make our evidence-chain contribution incremental rather than novel.

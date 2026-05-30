# Closest Prior Work Alignment

**Date**: 2026-05-27  
**Basis**: first-pass metadata/abstract review from `PAPER_TABLE.csv`. This file should be refined after PDF-level reading.

## Executive Summary

The current idea should not be framed as a generic V2X trust mechanism or a generic pose-error-robust cooperative perception model. Those areas are already populated.

The defensible gap is narrower:

```text
runtime object/message-level information availability
= source trust
+ current message usability
+ residual spatial correctability
+ downstream safety impact
```

The strongest angle is to preserve useful but biased cooperative information through explicit correction, while preventing high-impact abnormal messages from corrupting SafeCoDriver's safety constraint outputs.

## 1. V2X-DGPE: Domain Gaps and Pose Errors

**Overlap**

- Directly addresses pose errors and domain gaps in V2X collaborative perception.
- Uses historical information and deformable attention to handle feature misalignment.
- Evaluates on DAIR-V2X.

**Gap**

- It is a feature-level collaborative 3D detection framework.
- It does not appear, from abstract-level review, to output explicit message usability, source trust, evidence chains, or accept/correct/downweight/quarantine decisions.
- It handles pose error as a robustness problem inside fusion, not as a runtime correctability decision with downstream safety feedback.

**Impact on our positioning**

Avoid saying "we solve pose errors in V2X cooperative perception." Say:

> We estimate whether a spatially inconsistent cooperative message is correctable, apply an explicit object-level residual correction when possible, and validate the effect on downstream safety constraints.

## 2. V2X-ViT

**Overlap**

- Canonical robust V2X cooperative perception method.
- Handles asynchronous sharing, pose errors, and heterogeneous V2X components.
- Uses attention-based multi-agent fusion.

**Gap**

- Focuses on perception accuracy and robust representation/fusion.
- Does not provide a message-level trust/calibration layer for arbitrary object-level cooperative perception outputs.
- Does not explicitly reason about malicious/faulty message isolation or downstream collision-safety impact.

**Impact on our positioning**

V2X-ViT should be treated as a backbone/baseline family, not as a direct competitor to SafeCoDriver's object-level trust layer.

## 3. V2X Misbehavior and Collective Perception Service

**Overlap**

- Direct red flag for the trust/evidence-exchange part.
- Discusses authenticated but wrong V2X data, false location information, bogus objects, incorrect events, and Collective Perception Message security/standardization.
- Very close to the problem motivation.

**Gap**

- Standardization/security consideration rather than a downstream safety-calibrated perception correction algorithm.
- Does not appear to propose the specific object-level residual offset correction + SafeCoDriver safety-impact loop.

**Impact on our positioning**

Do not claim novelty for "misbehavior detection in collective perception" or "V2X evidence exchange" broadly. Use this as a motivating and threat-model source, then differentiate:

```text
standardization/misbehavior concern -> our runtime availability and correction module
```

## 4. V2X-INCOP

**Overlap**

- Handles communication interruption and missing cooperative information.
- Uses historical cooperative information to recover missing data.
- Explicitly motivated by safety risks from imperfect V2X communication.

**Gap**

- Mainly covers missing/interrupted messages.
- Does not target malicious forged objects, stable spatial misalignment, or mixed abnormal information under one availability framework.
- Recovery is predictive; our proposed correction is residual offset and trust based.

**Impact on our positioning**

V2X-INCOP is closest for "non-malicious missing information"; our method should include missing/dropout experiments but not overclaim there.

## 5. When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We?

**Overlap**

- Empirically studies V2X cooperative perception error patterns.
- Links increased cooperative perception errors to driving violations.
- Notes lack of robustness under online communication interference.

**Gap**

- Empirical study and diagnosis, not a correction/trust method.
- Provides strong motivation for measuring downstream safety effects.

**Impact on our positioning**

This is likely a key intro/related-work citation. It supports the claim that perception errors should be evaluated by driving impact, not only mAP.

## 6. Testing the Fault-Tolerance of Multi-Sensor Fusion Perception

**Overlap**

- Injects camera/LiDAR faults into multi-sensor fusion perception.
- Measures autonomous-driving system-level behavior and safety violations.
- Very relevant to anomaly/fault evaluation methodology.

**Gap**

- Not V2X cooperative perception.
- Does not handle cooperative message trust or inter-vehicle evidence exchange.

**Impact on our positioning**

Borrow fault models and downstream safety metric logic, but keep the contribution centered on V2X/cooperative message availability.

## 7. A Novel Probabilistic V2X Data Fusion Framework

**Overlap**

- Object/data-level V2X fusion for collective perception.
- Considers cross-correlation and probabilistic consistency.
- Demonstrates planning/decision-making benefits.

**Gap**

- More about consistent probabilistic fusion than abnormal source trust/correction.
- Need PDF-level check: it may already include uncertainty/correlation handling that overlaps with our residual/confidence formulation.

**Impact on our positioning**

This is important for object-level fusion baselines. Our method should not just be "weighted fusion"; it must include anomaly correctability and source-level trust update.

## 8. Selective Communication for Cooperative Perception

**Overlap**

- Chooses which cooperative vehicles to communicate with based on navigation-critical information.
- Safety-critical driving scenario simulations.

**Gap**

- Selection is about usefulness/bandwidth, not correctness/trust/correctability.
- Does not calibrate shifted or faulty information.

**Impact on our positioning**

Useful contrast:

```text
selective communication asks "who is useful to hear from?"
TrustCalib asks "given what we heard, is it usable, correctable, and safe to trust?"
```

## Implications For Method Design

The method should include these explicit components to stay differentiated:

1. **Residual offset estimate** at object/message level, not only learned feature robustness.
2. **Correctability score** that separates stable spatial bias from unstructured/fraudulent inconsistency.
3. **Action policy**: accept/correct/downweight/quarantine.
4. **Downstream safety impact** using SafeCoDriver outputs: V1 probability, geometric threats, WPC, min TTC, target speed factor.
5. **Evidence chain** that records matched objects, residuals, offset stability, and peer evidence.

## Claims To Avoid

Avoid broad claims:

- "first to handle pose errors in V2X cooperative perception"
- "first trust mechanism for V2X"
- "first misbehavior detection for collective perception"
- "first robust cooperative perception under communication problems"

Safer claims:

- "a safety-facing object-level availability and calibration layer"
- "correctability-aware handling of spatially biased cooperative messages"
- "downstream-safety-grounded trust update"
- "unified evaluation of attacks, faults, misalignment, and noise through usability/correctability"

## Next Reading Actions

1. PDF-level read of V2X-DGPE method section.
2. PDF-level read of V2X Misbehavior/CPS standardization paper.
3. PDF-level read of probabilistic V2X fusion framework.
4. Check whether any of them estimates explicit SE(2) residual offsets at runtime.
5. Check whether any uses downstream planner/safety-module deltas as part of trust update.

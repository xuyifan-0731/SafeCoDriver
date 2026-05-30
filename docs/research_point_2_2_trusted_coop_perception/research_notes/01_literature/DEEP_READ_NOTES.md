# PDF-Level Deep Read Notes

**Date**: 2026-05-27  
**Method**: Downloaded arXiv PDFs, extracted text with `pypdf`, scanned method/problem sections and targeted terms.  
**Scope**: This is a first PDF-level pass, not a full line-by-line related work survey.

## Papers Downloaded And Extracted

| arXiv ID | Paper |
|---|---|
| 2501.02363 | V2X-DGPE: Addressing Domain Gaps and Pose Errors for Robust Collaborative 3D Object Detection |
| 2112.02184 | V2X Misbehavior and Collective Perception Service: Considerations for Standardization |
| 2203.16964 | A Novel Probabilistic V2X Data Fusion Framework for Cooperative Perception |
| 2203.10638 | V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer |
| 2304.11821 | Interruption-Aware Cooperative Perception for V2X Communication-Aided Autonomous Driving |
| 2509.24927 | When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? |
| 2504.13420 | Testing the Fault-Tolerance of Multi-sensor Fusion Perception in Autonomous Driving Systems |

Text files are stored under:

```text
/raid/xuyifan/trusted_coop_perception/research/01_literature/text
```

## Q1: Does prior work already estimate explicit runtime SE(2) residual offsets?

### Finding

No direct match found in the first deep-read set.

### Evidence

**V2X-DGPE** is directly about pose errors, latency, GPS noise, and feature misalignment. However, its mechanism is feature-level: knowledge distillation, feature compensation, collaborative fusion, and deformable attention that dynamically offsets sampling points. The "offset" here refers to learned feature sampling offsets, not an explicit source-level `dx, dy, dtheta` correction emitted as a calibrated cooperative message.

**V2X-ViT** handles localization error and time delay by feature warping, delay-aware positional encoding, and multi-scale attention. Again, this is a fusion-backbone robustness mechanism, not a runtime object-level residual transform with an action policy.

**Probabilistic V2X Data Fusion** performs coordinate transform and covariance-intersection-based track fusion under unknown cross-correlation. It handles uncertainty consistency, not abnormal-source correctability or explicit source-offset estimation.

### Current Conclusion

The proposed `OffsetEstimate(dx, dy, dtheta)` and `correctable_score` remain plausible differentiators if implemented as explicit outputs and evaluated against injected known shifts.

## Q2: Does prior work already define evidence-chain fields for CPM trust/misbehavior?

### Finding

Partially yes. This is a novelty risk.

### Evidence

`V2X Misbehavior and Collective Perception Service` directly discusses:

- authenticated but wrong V2X data;
- false location information;
- bogus object reporting;
- CPM trustworthiness;
- misbehavior detectors and reporting;
- use of CAM/CPM consistency;
- misbehavior reports;
- time-offset attacks.

It also states that CPM misbehavior specification is not fully covered by current standards and proposes work items for CPS/CPM misbehavior protection.

### Current Conclusion

Do not claim novelty for "evidence exchange" or "V2X misbehavior detection" in general. In the proposed work, evidence exchange should be a support mechanism. The novelty should be tied to:

```text
object-level residual offset evidence
+ correctability
+ downstream safety impact
```

rather than generic V2X misbehavior reporting.

## Q3: Does prior work use downstream planner/safety deltas to update trust?

### Finding

No direct match found in the first deep-read set.

### Evidence

**Probabilistic V2X Data Fusion** demonstrates planning/decision-making benefits of cooperative perception. It does not appear to feed the planner's response back into a source-trust update.

**When Autonomous Vehicle Meets V2X Cooperative Perception** connects cooperative perception errors to driving violations and identifies localization/missing/miscorrection error patterns. It is diagnostic/empirical, not a trust-calibration method.

**FADE** injects sensor faults and discovers ADS safety violations. It uses system-level behavior to test fault tolerance, not to update runtime trust in cooperative messages.

**V2X-INCOP** discusses downstream cascade failures and safety risks under communication interruption, but its method recovers missing information through spatial-temporal prediction and distillation, not safety-delta trust updates.

### Current Conclusion

`safety_impact = delta(Hybrid stats with/without message)` is currently the strongest differentiator:

```text
delta collision_prob
delta min_ttc
delta n_geometric_threats
delta modification_rate
delta target_speed_factor
```

This should be made central in method and experiments.

## Paper-Specific Notes

### V2X-DGPE

**Covers**

- Domain gap between heterogeneous nodes.
- Pose errors from latency and GPS localization noise.
- Feature misalignment.
- Deformable attention sampling offsets.
- DAIR-V2X evaluation.

**Does Not Cover, Based On First Pass**

- Explicit per-source trust.
- Object/message-level accept/correct/downweight/quarantine.
- Runtime `dx, dy, dtheta` residual offset as an interpretable output.
- Downstream SafeCoDriver-like safety feedback.

**Implication**

Use as closest pose-error prior. Our method should be called an object-level safety calibration layer, not a pose-error robust fusion backbone.

### V2X Misbehavior and Collective Perception Service

**Covers**

- Authenticated but wrong data.
- False location and bogus objects.
- CPM security and standardization gaps.
- Misbehavior reporting and detectors.
- Sensor/fusion trustworthiness concerns.

**Does Not Cover, Based On First Pass**

- A concrete SafeCoDriver-like correction algorithm.
- Residual offset estimation and calibration of still-usable biased information.
- Downstream safety-metric feedback into trust.

**Implication**

This should be cited as threat-model/standardization motivation and as a boundary for novelty.

### Probabilistic V2X Data Fusion

**Covers**

- Object/track-level V2X data fusion.
- Unknown cross-correlation.
- Covariance intersection.
- Planning/decision-making safety benefits.

**Does Not Cover, Based On First Pass**

- Malicious/faulty source handling.
- Correctability scoring.
- Trust update.
- Explicit residual source calibration.

**Implication**

This is a relevant baseline family for uncertainty-aware object-level fusion. Our method must not collapse into "weighted fusion"; it needs abnormality and safety-impact logic.

### V2X-ViT

**Covers**

- V2X feature fusion.
- Localization error and time delay.
- Feature warping and delay-aware positional encoding.
- Robustness under noisy settings.

**Does Not Cover, Based On First Pass**

- Message-level trust/correctability.
- Evidence chain.
- Downstream safety feedback.

**Implication**

Good backbone prior and robustness baseline, but not a direct blocker.

### V2X-INCOP

**Covers**

- Communication interruption.
- Missing cooperative messages.
- Historical information recovery.
- Downstream risk motivation.

**Does Not Cover, Based On First Pass**

- Forged/bogus cooperative information.
- Stable residual spatial correction.
- Per-source trust and evidence exchange.

**Implication**

Use as missing-message prior. Our mixed-abnormality evaluation should include dropout but should not claim interruption recovery as main novelty.

### How Far Are We?

**Covers**

- Empirical cooperative perception error patterns.
- Localization errors, missing errors, miscorrected localization errors.
- Communication latency and pose error.
- Driving violations and odds of violation increase.

**Does Not Cover, Based On First Pass**

- Runtime repair method.
- Trust/correctability layer.

**Implication**

Strong motivation for using downstream safety metrics.

### FADE

**Covers**

- Camera/LiDAR fault models.
- Fault injection.
- System-level ADS safety violations.
- Differential fuzzing.

**Does Not Cover**

- V2X cooperative messages.
- Source trust or evidence exchange.
- Correcting still-usable abnormal cooperative information.

**Implication**

Borrow evaluation philosophy and fault taxonomy, not method.

## Updated Novelty Judgment

```text
Generic V2X trust/evidence exchange: weak novelty
Pose-error robust cooperative perception: weak novelty
Object-level correctability + explicit residual calibration: moderate novelty
Downstream-safety-impact-aware trust update: strongest novelty
```

## Recommended Method Constraint

The final method should include, at minimum:

1. An explicit `OffsetEstimate(dx, dy, dtheta)` or a documented reason for translation-only first version.
2. A `correctable_score` based on residual improvement, inlier ratio, offset stability, and covariance.
3. An action policy: accept / correct / downweight / quarantine.
4. A `safety_impact` term computed by SafeCoDriver outputs with and without the message.
5. Experiments showing correction beats hard filtering under stable spatial bias.

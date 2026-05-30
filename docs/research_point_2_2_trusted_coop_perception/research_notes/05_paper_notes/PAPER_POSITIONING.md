# Paper Positioning Notes

## Working Title Options

1. TrustCalib: Correctability-Aware Trust Calibration for Safe Cooperative Perception
2. From Anomaly Detection to Information Availability in V2X Cooperative Perception
3. Safety-Grounded Usability Evaluation for Spatially Misaligned Cooperative Perception

## Contribution Wording

### Strong Version

We propose an information availability framework for V2X cooperative perception that jointly estimates long-term source trust, instantaneous message usability, and residual spatial correctability. Unlike hard anomaly filters, the framework corrects stable spatially biased messages and only quarantines messages whose inconsistency is not explainable or whose downstream safety impact is high.

### Conservative Version

We introduce a lightweight pre-fusion calibration layer for object-level cooperative perception outputs. The layer estimates residual offsets, assigns usability scores, and feeds a corrected perception result into an existing safety constraint module. Experiments under injected spatial shifts, faults, and forged objects evaluate both perception consistency and downstream driving safety.

## Reviewer Questions To Prepare For

1. Is this just pose-error compensation?
2. Is this just trust management for VANETs?
3. Why not train a stronger cooperative perception backbone?
4. How do you know the anomaly labels are realistic?
5. Does the trust module hurt performance when the cooperative message is genuinely useful?
6. Is evidence exchange secure against collusion?

## Short Answers

1. Pose-error compensation is one branch; the key output is availability and correctability over mixed abnormal information.
2. VANET trust usually scores nodes/messages; this work ties trust to object-level residual calibration and downstream safety impact.
3. The method is intentionally model-agnostic and can wrap different backbones or GT object-level cooperative outputs.
4. Start with controlled injection for causal attribution, then validate with SUMO closed-loop stress cases.
5. Include Ego-Only/Hard-Filter/Raw-Coop/Oracle-Calib baselines to show the retention-vs-safety trade-off.
6. First version handles non-colluding or minority malicious sources; collusion is a limitation unless B5 is extended.

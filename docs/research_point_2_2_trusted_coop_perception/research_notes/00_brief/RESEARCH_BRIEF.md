# Research Brief

**Date**: 2026-05-27  
**Topic**: Spatial-misalignment-aware safe and trustworthy cooperative perception under abnormal information.  
**Target codebase**: SafeCoDriver at `/raid/xuyifan/jiqiuyu`.

## 1. Problem Anchor

Cooperative perception improves visibility by sharing information across vehicles or infrastructure, but the fusion stage can be polluted by:

- malicious attacks: forged objects, tampered positions, adversarially modified tracks;
- non-malicious failures: sensor faults, dropped objects, stale messages, calibration drift;
- spatial misalignment: pose error, time delay, residual coordinate transformation error;
- intrinsic uncertainty: noise, occlusion, detection confidence mismatch.

The current SafeCoDriver pipeline assumes that the cooperative perception input is already usable. This research point adds a trust and calibration layer before downstream safety correction.

## 2. Working Thesis

Abnormal cooperative information should not be handled as a binary normal/abnormal label. A safer formulation is:

```text
information availability = vehicle trust + current message usability + correctability
```

The method should decide:

```text
accept / correct / downweight / quarantine
```

and should output a calibrated `PerceptionResult` for existing SafeCoDriver modules.

## 3. Proposed Method Skeleton

```text
Raw cooperative messages
  -> multi-source object association
  -> residual SE(2) offset estimation
  -> correctability scoring
  -> current message usability scoring
  -> long-term vehicle trust update
  -> minimal evidence exchange
  -> calibrated cooperative perception
  -> SafeCoDriver HybridSafetyConstraint
```

## 4. Primary Claims To Test

**C1: Unified availability evaluation.**  
A single availability framework can handle pose error, sensor faults, missing objects, forged objects, and noise better than separate hard filters.

**C2: Correction beats discard when the anomaly is stable spatial misalignment.**  
Stable residual offsets should be corrected and retained, because hard filtering discards useful blind-spot information.

**C3: Downstream safety impact is the right validation target.**  
The method should be evaluated not only by perception mAP/residuals, but also by SafeCoDriver metrics: WPC%, FA(f), Det(s), SUMO CollRate, hard braking, and false avoidance.

## 5. Immediate Research Questions

1. What prior work already handles pose error in V2X cooperative perception?
2. What prior work handles malicious or faulty V2X messages?
3. Do existing methods unify pose error, attack, faults, and noise into one usability/correctability framework?
4. Which public datasets and simulation knobs can support anomaly injection?
5. What is the simplest implementation that can produce credible first results on DeepAccident and SUMO?

## 6. First-Pass Positioning

Avoid claiming a new cooperative perception backbone. The contribution should be framed as a **safety-facing trust calibration layer** that is:

- model-agnostic: can wrap object-level outputs from any cooperative perception system;
- correction-aware: estimates residual offsets instead of discarding all abnormal messages;
- evidence-based: produces explicit evidence chains;
- downstream-grounded: validates by planning/safety effects, not only detection metrics.

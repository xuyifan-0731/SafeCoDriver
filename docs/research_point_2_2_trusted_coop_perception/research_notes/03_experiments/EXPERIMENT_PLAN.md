# Experiment Plan

**Problem**: Cooperative perception messages can be shifted, faulty, malicious, or noisy; downstream SafeCoDriver currently treats them as already usable.  
**Method thesis**: A trust-calibration layer that estimates message usability and residual offset can preserve useful abnormal information while protecting downstream safety.  
**Date**: 2026-05-27

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Correctable spatial misalignment should be corrected, not discarded. | This is the main novelty over hard anomaly filtering. | Under injected shifts, TrustCalib approaches Oracle-Calib and beats Raw-Coop/Hard-Filter on WPC% and residuals. | B1, B2 |
| C2: Safety-impact-aware availability improves downstream safety. | Connects research point 2.2 to SafeCoDriver. | Forged high-impact obstacles cause fewer false brakes/false avoidances than Raw-Coop, without hurting real collision detection. | B3, B4 |
| C3: Evidence exchange improves reliability when ego evidence is partial. | Matches the user's collaborative trust mechanism. | Multi-source synthetic SUMO evidence improves action accuracy and trust calibration. | B5 |

## Experiment Blocks

### B0: Sanity and Metric Reproduction

- **Claim tested**: none; ensure no regression.
- **Dataset/task**: current DeepAccident validation and SUMO key scenarios.
- **Runs**:
  - current `run_deepaccident_unified_metrics.py`
  - current `run_modified_sumo_comparison.py --scenario-set base/stress --key-methods`
- **Success criterion**: reproduce current baseline numbers within expected noise.
- **Priority**: MUST-RUN.

### B1: Spatial Shift Correction on DeepAccident

- **Claim tested**: C1.
- **Anomalies**:
  - `dx,dy`: `(0.5,0.5)`, `(2.0,1.0)`, `(4.0,2.0)` meters.
  - `dtheta`: `0`, `3`, `6` degrees if rotation is implemented.
- **Compared systems**:
  - Ego-Only
  - Raw-Coop
  - Hard-Filter
  - Trust-Only
  - Calib-Only
  - TrustCalib
  - Oracle-Calib
- **Metrics**:
  - Offset-MAE
  - residual before/after
  - correctability F1
  - WPC%, FA(f), Det(s), Mod%
- **Success criterion**: TrustCalib reduces residuals and WPC% significantly versus Raw-Coop, while retaining better Det(s) than Ego-Only/Hard-Filter.
- **Priority**: MUST-RUN.

### B2: Mixed Abnormal Information Taxonomy

- **Claim tested**: C1.
- **Anomalies**:
  - random Gaussian noise;
  - dropped objects;
  - stale message/time delay;
  - forged object on planned path;
  - non-physical position/velocity jump.
- **Metrics**:
  - usability AUC;
  - recommended action accuracy;
  - quarantine rate;
  - WPC%, FA(f), Det(s).
- **Success criterion**: stable shifts are mostly `correct`; forged/jumpy high-risk messages are mostly `downweight/quarantine`; noisy low-risk messages are not over-penalized.
- **Priority**: MUST-RUN.

### B3: Safety-Impact Trust Update Ablation

- **Claim tested**: C2.
- **Compared variants**:
  - TrustCalib without downstream impact;
  - TrustCalib with downstream impact from Hybrid stats;
  - overly conservative variant that quarantines all low-consensus messages.
- **Metrics**:
  - false alarm frame rate;
  - target speed factor distribution;
  - hard brake rate;
  - WPC%;
  - trust ECE.
- **Success criterion**: safety-impact-aware update reduces false high-risk responses to forged messages without suppressing genuine high-risk blind-spot objects.
- **Priority**: MUST-RUN after B1.

### B4: SUMO Closed-Loop Attack and Fault Scenarios

- **Claim tested**: C2.
- **Scenario types**:
  - fake obstacle ahead causing unnecessary braking;
  - shifted cooperative attacker position;
  - missing cooperative attacker;
  - rear-risk stress with false front obstacle.
- **Compared systems**:
  - Raw-Coop + Hybrid;
  - Ego-Only + Hybrid;
  - TrustCalib + Hybrid;
  - TrustCalib + RearEscape.
- **Metrics**:
  - CollRate;
  - second collision;
  - severity;
  - hard brake rate;
  - false avoid rate;
  - min TTC distribution.
- **Success criterion**: lower CollRate/false avoid/hard brake than Raw-Coop under abnormal messages, while maintaining low second collisions.
- **Priority**: SHOULD-RUN.

### B5: Evidence Exchange Simulation

- **Claim tested**: C3.
- **Setup**:
  - synthesize 3-5 sources in SUMO or DeepAccident-like object lists;
  - one malicious/faulty source;
  - partial views and different source trusts.
- **Compared systems**:
  - local-only availability;
  - naive vote;
  - trust-weighted evidence exchange;
  - oracle source labels.
- **Metrics**:
  - source classification F1;
  - trust calibration ECE;
  - communication bytes/frame;
  - action accuracy.
- **Success criterion**: trust-weighted evidence exchange beats naive voting and local-only under partial observability.
- **Priority**: NICE-TO-HAVE for first paper, useful for extension.

## Run Order

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Reproduce current baseline | B0 | Existing metrics reproduce | Low | Environment/data path issues |
| M1 | Build anomaly injection | B1 small subset | Known offset recovered within 0.5m | Low | Object association ambiguity |
| M2 | Full DeepAccident shift study | B1 full val | TrustCalib beats Raw-Coop/Hard-Filter | Medium | Synthetic shifts too easy |
| M3 | Mixed anomalies | B2 | Actions match anomaly taxonomy | Medium | Ground-truth action labels debatable |
| M4 | Safety-impact ablation | B3 | FA/hard brake reduced | Medium | Impact metric too noisy |
| M5 | SUMO closed-loop | B4 | Closed-loop safety improves | Medium | Scenario engineering time |
| M6 | Evidence exchange | B5 | Consensus helps | Low/Medium | Needs convincing multi-source setup |

## Minimal Implementation Needed

```text
coop_safety/trust/
  association.py
  offset_estimator.py
  usability.py
  trust_manager.py
  perception_calibrator.py

experiments/
  anomaly_injection.py
  run_trust_calib_deepaccident.py
  run_trust_calib_sumo.py
```

## First Stop/Go Criterion

Before building the full trust framework, run a small DeepAccident pilot:

```text
20 validation scenarios
medium shift: dx=2m, dy=1m
compare Raw-Coop, Hard-Filter, Calib-Only, Oracle-Calib
```

If object-level calibration cannot recover WPC% meaningfully, the method needs rethinking before adding trust/evidence complexity.

# Literature Triage

**Generated**: 2026-05-27  
**Input**: `PAPER_TABLE.md` and first-pass seed review.

## Must Deep-Read

| Priority | Paper | Reason |
|---:|---|---|
| 1 | V2X-DGPE: Addressing Domain Gaps and Pose Errors for Robust Collaborative 3D Object Detection | Closest pose-error / spatial misalignment prior. Need determine whether it estimates explicit offsets or only learns robust attention. |
| 2 | V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer | Canonical V2X robust fusion baseline; handles pose error/asynchrony/heterogeneity. |
| 3 | Interruption-Aware Cooperative Perception for V2X Communication-Aided Autonomous Driving | Closest missing-message / communication failure prior. |
| 4 | When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? | Empirical error-pattern and driving-violation motivation; likely important for intro and problem framing. |
| 5 | Testing the Fault-Tolerance of Multi-Sensor Fusion Perception in Autonomous Driving Systems | Useful for fault injection and downstream behavior metrics, even if not V2X-specific. |
| 6 | A Novel Probabilistic V2X Data Fusion Framework for Cooperative Perception | Potentially relevant to uncertainty-aware object-level fusion. |
| 7 | Selective Communication for Cooperative Perception in End-to-End Autonomous Driving | Usefulness-aware selection; compare against our usability/trust concept. |

## Important Background

| Paper | Use |
|---|---|
| Collaborative Perception in Autonomous Driving: Methods, Datasets and Challenges | Survey/taxonomy and dataset coverage. |
| UniE2EV2X | DeepAccident baseline and safety/end-to-end framing. |
| V2X-Real | Future real-world V2X evaluation option. |
| V2X-DG | Domain generalization contrast. |
| HeCoFuse | Heterogeneous sensor misalignment contrast. |
| CooPre | Pretraining/robust representation contrast. |
| NLOS Dies Twice | Communication/perception coupling motivation. |

## Likely Exclude Or Cite Briefly

| Paper Type | Reason |
|---|---|
| General AV surveys | Too broad unless used for one sentence. |
| 6G / vehicular communication surveys | Useful only for V2X reliability background. |
| Standalone VLM/camera adversarial attacks | Not cooperative perception; use only for attack taxonomy. |
| Quantum / generic Byzantine sensor fusion | Too far from autonomous driving; maybe cite as theory background if needed. |

## Open Novelty Checks

1. Search and deep-read specifically for **trust-aware cooperative perception**.
2. Search for **misbehavior detection in collective perception / CPM / ETSI ITS**.
3. Check whether any paper already outputs **accept/correct/downweight/quarantine** style decisions.
4. Check whether object-level V2X fusion papers already estimate a residual SE(2) transform at runtime.
5. Check if downstream safety metrics have been used for cooperative perception abnormality handling.

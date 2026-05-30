# Seed Papers

This is a first-pass seed list from ARIS helper searches on 2026-05-27. It is intentionally conservative: papers are grouped by how they affect the current research point.

## A. Cooperative Perception Foundations and Benchmarks

| Paper | Year | Why It Matters | Gap For This Work |
|---|---:|---|---|
| V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer | 2022 | Handles heterogeneous V2X fusion and explicitly studies asynchronous sharing, pose errors, and harsh/noisy settings. | Robust backbone, but not a unified information availability / trust / correctability layer. |
| Collaborative Perception in Autonomous Driving: Methods, Datasets and Challenges | 2023 | Survey of cooperative perception methods, datasets, ideal scenarios, and real-world issues. | Use as related-work backbone; need extract its taxonomy of real-world issues. |
| V2X-Real: a Large-Scale Dataset for Vehicle-to-Everything Cooperative Perception | 2024 | Real-world V2X dataset with vehicle/infrastructure cooperation and public benchmarks. | Candidate future dataset for object-level trust calibration beyond DeepAccident/SUMO. |
| UniE2EV2X: Unified End-to-End V2X Cooperative Autonomous Driving | 2024 | DeepAccident-relevant V2X end-to-end autonomous driving baseline; already appears in SafeCoDriver comparisons. | Focuses on unified driving network, not message-level availability or correction. |
| Research Challenges and Progress in the End-to-End V2X Cooperative Autonomous Driving Competition | 2025 | Summarizes practical V2X constraints: bandwidth-aware fusion, robust multi-agent planning, heterogeneous integration. | Good positioning source for safety/downstream evaluation. |

## B. Pose Error, Spatial Misalignment, and Domain Gap

| Paper | Year | Why It Matters | Gap For This Work |
|---|---:|---|---|
| V2X-DGPE: Addressing Domain Gaps and Pose Errors for Robust Collaborative 3D Object Detection | 2025 | Directly targets domain gaps and pose errors; uses historical information, deformable attention, and DAIR-V2X. | Learns a robust feature fusion model; does not output explicit usability, evidence chains, or residual calibration decisions. |
| V2X-DG: Domain Generalization for Vehicle-to-Everything Cooperative Perception | 2025 | Studies domain generalization across OPV2V, V2XSet, V2V4Real, DAIR-V2X. | Domain-level robustness, not per-message trust/correctability. |
| HeCoFuse: Cross-Modal Complementary V2X Cooperative Perception with Heterogeneous Sensors | 2025 | Addresses cross-modality feature misalignment and heterogeneous sensor setups. | Strong baseline for heterogeneity; not attack/fault/trust oriented. |
| CooPre: Cooperative Pretraining for V2X Cooperative Perception | 2024 | Uses self-supervised pretraining to improve multi-agent spatial correlations and robustness. | Pretraining improves representation; does not solve runtime abnormal message handling. |

## C. Communication Interruption and Selective Cooperation

| Paper | Year | Why It Matters | Gap For This Work |
|---|---:|---|---|
| V2X-INCOP: Interruption-Aware Cooperative Perception for V2X Communication-Aided Autonomous Driving | 2023 | Handles message interruption using historical cooperation and spatial-temporal prediction. | Covers missing messages, not malicious/faulty/spatially biased messages in one framework. |
| Selective Communication for Cooperative Perception in End-to-End Autonomous Driving | 2023 | Selects cooperative vehicles carrying information critical to navigation planning. | Selection is based on usefulness/communication, not trustworthiness and correction. |
| NLOS Dies Twice: Challenges and Solutions of V2X for Cooperative Perception | 2023 | Connects perception blind zones and V2X communication NLOS problems. | Communication reliability perspective; useful for evidence-exchange constraints. |

## D. Faults, Attacks, and Byzantine/Fault-Tolerant Fusion

| Paper | Year | Why It Matters | Gap For This Work |
|---|---:|---|---|
| Testing the Fault-Tolerance of Multi-Sensor Fusion Perception in Autonomous Driving Systems | 2025 | Systematically injects sensor faults and measures autonomous-driving system-level behaviors. | Not V2X-specific, but important for fault models and downstream safety metrics. |
| Capacity of Cooperative Fusion in the Presence of Byzantine Sensors | 2006 | Classic information-theoretic result for Byzantine sensor fusion. | Not autonomous driving, but useful for threat model and trust/fusion limits. |
| PG-Attack: Precision-Guided Adversarial Attack Framework Against Vision Foundation Models for Autonomous Driving | 2024 | Shows perception attacks can affect autonomous-driving reliability and safety. | Standalone VFM/camera attack, not cooperative message trust, but informs malicious taxonomy. |

## E. Empirical Error and Safety Motivation

| Paper | Year | Why It Matters | Gap For This Work |
|---|---:|---|---|
| When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? | 2025 | Empirical study identifying cooperative perception error patterns and linking increased errors to driving violations and online communication interference. | Very close motivation; need verify details and position this work as a repair/calibration method. |

## First-Pass Gap Summary

The closest works cover pieces of the problem:

- pose-error robustness: V2X-ViT, V2X-DGPE;
- interruption/missing data: V2X-INCOP;
- heterogeneous misalignment: HeCoFuse;
- fault injection and safety impact: FADE-style MSF fault testing;
- empirical V2X error taxonomy: 2025 "How Far Are We?" study.

The apparent gap is a runtime, object/message-level layer that unifies:

```text
long-term source trust
+ current message usability
+ residual offset correctability
+ minimal evidence exchange
+ downstream safety impact
```

This gap needs a deeper novelty check against trust-management and Byzantine vehicular-network literature.

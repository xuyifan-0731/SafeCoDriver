# Trust Update Formulation

**Topic**: vehicle trust update for object-level cooperative perception availability  
**Prototype**: `/raid/xuyifan/trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py`

## Motivation

Vehicle trust should not be a static prior and should not override object evidence unconditionally. The prototype shows the useful closed loop:

```text
object evidence -> safety-impact anomaly -> sender trust update -> future fusion policy
```

This document formalizes the prototype rule into a proposal-ready equation.

## Variables

For sender `i`, object `j`, frame `t`:

```text
T_i(t)      vehicle trust in [0,1]
C_i(t)      message correctability score
S_ij(t)     object support score from peer evidence
W_ij(t)     trust-weighted object support
U_ij(t)     unsupported indicator
I_ij(t)     downstream safety-impact score
P_ij(t)     probation state / unsupported age
```

Object support:

```text
S_ij(t) = number of calibrated peer messages matching object j
W_ij(t) = sum_k T_k(t) * 1[peer k supports object j]
```

Unsupported indicator:

```text
U_ij(t) = 1[S_ij(t) < tau_count or W_ij(t) < tau_weight]
```

Safety impact can combine geometric and path-level margins:

```text
I_ij(t) = alpha_g * geom_delta_ij
        + alpha_m * mod_delta_ij
        + alpha_p * max(0, -path_collision_margin_ij)
```

The current prototype records:

```text
geom_delta
mod_delta
path_min_distance
path_collision_margin
path_risk_step
```

## Trust Update

A bounded additive update:

```text
T_i(t+1) = clip(
  T_i(t)
  - lambda_u * A_i(t)
  - lambda_o * O_i(t)
  - lambda_d * D_i(t)
  + lambda_r * R_i(t),
  0, 1
)
```

where:

```text
A_i(t) = sum_j U_ij(t) * 1[I_ij(t) > tau_impact]
O_i(t) = offset_instability_i(t)
D_i(t) = peer_disagreement_i(t)
R_i(t) = sustained_consistency_i(t)
```

Prototype simplification:

```text
A_i(t) = number of high-impact unsupported objects in frame t
O_i(t) = max(0, residual_after_i(t) - tau_offset)
D_i(t) = max(0, peer_cluster_cov_i(t) - tau_disagree)
R_i(t) = 1 if no high-impact unsupported object and at least one supported object exists else 0
```

Prototype parameters:

```text
lambda_u = 0.2
lambda_r = 0.02
tau_impact = geom_delta > 2 or mod_delta > 0.2
tau_offset = 0.5
tau_disagree = 1.0
```

## Probation Policy

Trust can grant temporary probation only when the object risk is bounded:

```text
if U_ij(t)=1 and T_i(t) >= tau_trust and I_ij(t) <= tau_low_impact:
  action = probation_keep
elif U_ij(t)=1 and I_ij(t) > tau_low_impact:
  action = object_quarantine
else:
  action = keep
```

Availability-first ablation:

```text
if U_ij(t)=1 and T_i(t) >= tau_trust and allow_high_impact_probation:
  action = probation_keep
```

The ablation can recover objects under peer dropout, but it is unsafe under high-trust sender compromise unless trust dynamics quickly removes the sender's probation privilege.

## Prototype Evidence

Fake-front with high-impact probation override and trust dynamics:

```text
PrimaryTrustCalib: WPC 2.17%, warn 100.00%
MultiPeerObjectGuard + trust dynamics: WPC 0.33%, warn 52.00%
fake removal 85.2%, probation restore 9.5%
```

Example trust trajectory:

```text
frame 0: T=1.0 -> 0.8, high-impact fake gets temporary probation
frame 1: T=0.8 -> 0.6, fake gets temporary probation
frame 2: T=0.6 -> 0.4, fake is quarantined
```

Offset-instability check:

```text
correctable severe shift: residual_after ~= 0.054, trust remains 1.0
noise: residual_after ~= 1.056, trust decays to 0.0 under offset penalty
stale: residual_after ~= 1.227, trust decays to 0.0 under offset penalty
```

## Paper Claim

The method estimates cooperative information availability as a dynamic property:

```text
availability = vehicle trust
             + message correctability
             + object evidence support
             + downstream safety impact
             + temporal probation state
```

Unlike static trust filtering, the sender trust is updated by object-level evidence and downstream safety impact. Unlike whole-message quarantine, the method preserves supported cooperative objects while suppressing unsupported high-impact objects.

# Consensus Smoothing Final Update

**Date**: 2026-05-28  
**Prototype**: `/raid/xuyifan/trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py`  
**Environment**: `Android-Lab`

## Mechanism Update

The final prototype now adds an object-level peer consensus stage:

```text
TrustCalib + TimeCalib + MultiPeerObjectGuard + BoxGuard
+ PeerConsensusSmoothing + MissingRecovery
```

The main change is for noisy but still useful cooperative messages. The previous method removed unsupported fake objects well, but noisy real-object positions still caused elevated WPC in `noise+fake_front`. The updated policy uses calibrated support peers to smooth surviving cooperative objects and allows MissingRecovery from `downweight` / `quarantine` primary messages only when the candidate object has sufficient multi-peer evidence.

Recommended command shape:

```bash
python prototype/run_deepaccident_multipeer_pilot.py \
  --support-modes clean,shift \
  --enable-time-calib \
  --enable-missing-recovery \
  --missing-recovery-primary-actions accept,correct,time_correct,downweight,quarantine \
  --enable-box-margin-guard \
  --enable-peer-consensus-smoothing
```

Important safeguards:

- ObjectGuard and BoxGuard run before smoothing, so high-impact unsupported objects are filtered before they can be pulled into a peer consensus.
- MissingRecovery still requires `missing_min_peer_support=2` and `missing_min_trust_support=1.0` by default.
- PeerConsensusSmoothing uses a separate support gate: `smooth_min_peer_support=2`, `smooth_min_trust_support=1.0`, `smooth_match_dist=4.0`, `smooth_max_cluster_cov=4.0`.

## Key Results

All rows below are `MultiPeerObjectGuard`, which is the final method row.

### 20x20 Fast, V1 Disabled

Output: `/raid/xuyifan/trusted_coop_perception/results/final_consensus_policy_20x20_fast/summary.csv`

| Mode | WPC | Warn | wp_coll/wp_total | Avg Smoothed | Avg Missing Recovered | Fake Removal |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.425% | 46.50% | 17/4000 | 0.00/fr | 0.00/fr | 0.0% |
| shift | 0.425% | 46.50% | 17/4000 | 0.00/fr | 0.00/fr | 0.0% |
| stale | 0.425% | 46.50% | 17/4000 | 0.07/fr | 0.02/fr | 0.0% |
| drop | 0.425% | 46.50% | 17/4000 | 0.00/fr | 11.66/fr | 0.0% |
| fake_front | 0.425% | 46.50% | 17/4000 | 0.00/fr | 0.00/fr | 100.0% |
| shift+drop | 0.425% | 46.50% | 17/4000 | 0.00/fr | 11.66/fr | 0.0% |
| stale+drop | 0.425% | 46.50% | 17/4000 | 0.03/fr | 11.66/fr | 0.0% |
| noise+fake_front | 0.425% | 46.50% | 17/4000 | 13.04/fr | 4.30/fr | 99.4% |

### 20x20, V1 Enabled

Output: `/raid/xuyifan/trusted_coop_perception/results/final_consensus_policy_v1_20x20/summary.csv`

| Mode | WPC | Warn | wp_coll/wp_total | Avg Smoothed | Avg Missing Recovered | Fake Removal |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.425% | 56.50% | 17/4000 | 0.00/fr | 0.00/fr | 0.0% |
| shift | 0.425% | 57.50% | 17/4000 | 0.00/fr | 0.00/fr | 0.0% |
| stale | 0.425% | 56.50% | 17/4000 | 0.07/fr | 0.02/fr | 0.0% |
| drop | 0.425% | 57.00% | 17/4000 | 0.00/fr | 11.66/fr | 0.0% |
| fake_front | 0.425% | 56.50% | 17/4000 | 0.00/fr | 0.00/fr | 100.0% |
| shift+drop | 0.425% | 57.75% | 17/4000 | 0.00/fr | 11.66/fr | 0.0% |
| stale+drop | 0.425% | 57.00% | 17/4000 | 0.03/fr | 11.66/fr | 0.0% |
| noise+fake_front | 0.425% | 57.00% | 17/4000 | 13.04/fr | 4.30/fr | 99.4% |

### Full Validation Frames, V1 Disabled

Output: `/raid/xuyifan/trusted_coop_perception/results/final_consensus_policy_full_val_fast/summary.csv`

Subset: 22 checkpoint validation scenarios, 1753 frames, 17530 waypoints.

| Mode | WPC | Warn | wp_coll/wp_total | Avg Smoothed | Avg Missing Recovered | Fake Removal |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.291% | 43.87% | 51/17530 | 0.00/fr | 0.00/fr | 0.0% |
| shift | 0.291% | 43.87% | 51/17530 | 0.00/fr | 0.00/fr | 0.0% |
| stale | 0.291% | 43.87% | 51/17530 | 0.12/fr | 0.13/fr | 0.0% |
| drop | 0.291% | 43.87% | 51/17530 | 0.00/fr | 9.90/fr | 0.0% |
| fake_front | 0.291% | 43.87% | 51/17530 | 0.00/fr | 0.00/fr | 99.8% |
| shift+drop | 0.291% | 43.87% | 51/17530 | 0.00/fr | 9.90/fr | 0.0% |
| stale+drop | 0.291% | 43.87% | 51/17530 | 0.04/fr | 9.93/fr | 0.0% |
| noise+fake_front | 0.291% | 43.92% | 51/17530 | 10.81/fr | 3.85/fr | 99.0% |

### Accident Window, V1 Enabled

Output: `/raid/xuyifan/trusted_coop_perception/results/final_consensus_policy_v1_accident_window/summary.csv`

Subset: accident-centered windows, 3300 waypoints.

| Mode | WPC | Warn | wp_coll/wp_total | Avg Smoothed | Avg Missing Recovered | Fake Removal |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.606% | 74.24% | 20/3300 | 0.00/fr | 0.00/fr | 0.0% |
| shift | 0.606% | 74.24% | 20/3300 | 0.00/fr | 0.00/fr | 0.0% |
| stale | 0.606% | 74.24% | 20/3300 | 0.07/fr | 0.06/fr | 0.0% |
| drop | 0.606% | 73.94% | 20/3300 | 0.00/fr | 10.59/fr | 0.0% |
| fake_front | 0.606% | 74.24% | 20/3300 | 0.01/fr | 0.00/fr | 99.1% |
| shift+drop | 0.606% | 73.94% | 20/3300 | 0.00/fr | 10.59/fr | 0.0% |
| stale+drop | 0.606% | 73.94% | 20/3300 | 0.01/fr | 10.62/fr | 0.0% |
| noise+fake_front | 0.606% | 74.24% | 20/3300 | 11.95/fr | 3.85/fr | 98.5% |

## Real Cooperative Label Audit

Output: `/raid/xuyifan/trusted_coop_perception/results/real_coop_availability_audit/summary.json`

The DeepAccident loader can read `other_vehicle/label` for all 1753 validation frames:

- Coverage: 1753/1753 frames.
- Average ego-label objects: 35.68/frame.
- Average other_vehicle-label objects: 35.32/frame.
- Direct ego-label vs other_vehicle-label association residual: 20.95 m mean, 18.64 m median.

Interpretation: the available `other_vehicle` labels are not currently ego-frame aligned cooperative messages. They appear to be in the other vehicle's own frame under the current loader. Therefore, the current final results should still be claimed as synthetic object-list perturbation pilots with GT-derived support peers, not full real multi-vehicle validation.

## Updated Claim

Safe claim:

```text
In synthetic DeepAccident object-list perturbation pilots with reliable
GT-derived support peers, the final prototype recovers clean-level WPC for
spatial shift, stale delay, dropout, fake-front injection, shift/stale+drop,
and noise+fake-front compound anomalies. The noise boundary is handled by
evidence-gated peer consensus smoothing plus support-gated recovery from
downweighted/quarantined primary messages.
```

Remaining limitations:

- Support peers are still synthetic and GT-derived.
- Real `other_vehicle` labels need coordinate transformation before they can be used as ego-frame cooperative messages.
- Full fake collusion can still bypass count-only support unless sender trust is low or evidence chains are independently verified.
- Full-frame all-mode final run took about 13 minutes; caching candidate impact evaluations would be useful for sweeps.

| comparison | observed diff (percentage points) | scenario-bootstrap 95% CI (percentage points) | P(diff < 0) |
| --- | ---: | ---: | ---: |
| RealMultiEvidenceGuard - EgoOnly | -0.225 | [-1.075, 0.450] | 0.681 |
| RealMultiEvidenceGuard - RealPrimaryTrustCalib | -0.100 | [-0.300, 0.000] | 0.642 |
| RealMultiEvidenceGuard - CleanCoop oracle | 1.200 | [0.000, 3.525] | 0.000 |

Path-risk-aware update (`results/realmultisource_20x20_pathrisk_min2_thr0/frame_diagnostics.csv`):

| comparison | observed diff (percentage points) | scenario-bootstrap 95% CI (percentage points) | P(diff < 0) |
| --- | ---: | ---: | ---: |
| PathRisk RealMultiEvidenceGuard - EgoOnly | -0.300 | [-1.175, 0.400] | 0.753 |
| PathRisk RealMultiEvidenceGuard - RealPrimaryTrustCalib | -0.175 | [-0.450, 0.000] | 0.880 |
| PathRisk RealMultiEvidenceGuard - CleanCoop oracle | 1.125 | [0.000, 3.375] | 0.000 |

Notes:

- Bootstrap unit: validation scenario.
- Samples: 10,000 resamples with replacement over the 20 scenarios in `results/realmultisource_20x20_fast_min2_self_filtered_diag/frame_diagnostics.csv`.
- Negative diff means the first method has lower WPC.
- Interpretation: real multi-source evidence is directionally better than EgoOnly and RealPrimaryTrustCalib on this 20x20 subset, and path-risk-aware admission strengthens the direction. The improvement is still not a strong real-data superiority claim because the scenario-level confidence interval overlaps zero. The gap to CleanCoop oracle remains clear.

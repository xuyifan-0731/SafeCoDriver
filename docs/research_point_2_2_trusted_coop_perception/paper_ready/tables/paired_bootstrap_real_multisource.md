| comparison | observed diff (percentage points) | scenario-bootstrap 95% CI (percentage points) | P(diff < 0) |
| --- | ---: | ---: | ---: |
| RealMultiEvidenceGuard - EgoOnly | -0.225 | [-1.075, 0.450] | 0.681 |
| RealMultiEvidenceGuard - RealPrimaryTrustCalib | -0.100 | [-0.300, 0.000] | 0.642 |
| RealMultiEvidenceGuard - CleanCoop oracle | 1.200 | [0.000, 3.525] | 0.000 |

Notes:

- Bootstrap unit: validation scenario.
- Samples: 10,000 resamples with replacement over the 20 scenarios in `results/realmultisource_20x20_fast_min2_self_filtered_diag/frame_diagnostics.csv`.
- Negative diff means the first method has lower WPC.
- Interpretation: real multi-source evidence is directionally better than EgoOnly and RealPrimaryTrustCalib on this 20x20 subset, but the improvement is not yet statistically strong. The gap to CleanCoop oracle remains clear.

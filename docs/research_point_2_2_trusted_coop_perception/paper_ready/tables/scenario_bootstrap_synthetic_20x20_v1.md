| mode | comparison | observed diff (pp) | scenario-bootstrap 95% CI (pp) | P(diff < 0) | scenarios |
| --- | --- | ---: | ---: | ---: | ---: |
| drop | Final - Raw fusion | -0.875 | [-2.525, 0.250] | 0.898 | 20 |
| drop | Final - TrustCalib only | -0.875 | [-2.525, 0.250] | 0.898 | 20 |
| drop | Final - w/o MissingRecovery | -0.875 | [-2.525, 0.250] | 0.898 | 20 |
| fake_front | Final - Raw fusion | -1.750 | [-3.551, -0.400] | 0.998 | 20 |
| fake_front | Final - TrustCalib only | -1.750 | [-3.551, -0.400] | 0.998 | 20 |
| fake_front | Final - w/o BoxGuard | 0.025 | [0.000, 0.075] | 0.000 | 20 |
| noise+fake_front | Final - Raw fusion | -1.500 | [-2.650, -0.500] | 0.999 | 20 |
| noise+fake_front | Final - TrustCalib only | -2.500 | [-4.025, -1.250] | 1.000 | 20 |
| noise+fake_front | Final - w/o BoxGuard | -2.000 | [-3.175, -1.075] | 1.000 | 20 |
| noise+fake_front | Final - w/o MissingRecovery | -0.275 | [-0.825, 0.125] | 0.862 | 20 |
| noise+fake_front | Final - w/o smoothing | -0.025 | [-0.250, 0.175] | 0.535 | 20 |
| noise+fake_front | Final - w/o noisy recovery | -0.275 | [-0.825, 0.125] | 0.862 | 20 |

Notes:

- Bootstrap unit is scenario, with 10,000 resamples over the 20 checkpoint validation scenarios.
- Negative diff means `Final` has lower WPC than the compared baseline.
- This table turns the module ablation into a scenario-level statistical comparison.

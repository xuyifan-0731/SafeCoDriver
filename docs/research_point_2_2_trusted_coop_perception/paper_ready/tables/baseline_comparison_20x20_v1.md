| mode | variant | WPC | Warn | wp_coll/wp_total | fake removal | missing recovered/frame | smoothed/frame |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | EgoOnly | 1.850% | 53.75% | 74/4000 | NA | 0.00 | 0.00 |
| clean | CleanCoop oracle | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | Raw fusion | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | TrustCalib only | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | w/o MissingRecovery | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | w/o BoxGuard | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | w/o smoothing | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | w/o noisy recovery | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| clean | Final | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| drop | EgoOnly | 1.850% | 53.75% | 74/4000 | NA | 0.00 | 0.00 |
| drop | CleanCoop oracle | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| drop | Raw fusion | 1.300% | 53.25% | 52/4000 | NA | 0.00 | 0.00 |
| drop | TrustCalib only | 1.300% | 53.25% | 52/4000 | NA | 0.00 | 0.00 |
| drop | w/o MissingRecovery | 1.300% | 53.25% | 52/4000 | NA | 0.00 | 0.00 |
| drop | w/o BoxGuard | 0.425% | 57.00% | 17/4000 | NA | 11.66 | 0.00 |
| drop | w/o smoothing | 0.425% | 57.00% | 17/4000 | NA | 11.66 | 0.00 |
| drop | w/o noisy recovery | 0.425% | 57.00% | 17/4000 | NA | 11.66 | 0.00 |
| drop | Final | 0.425% | 57.00% | 17/4000 | NA | 11.66 | 0.00 |
| fake_front | EgoOnly | 1.850% | 53.75% | 74/4000 | NA | 0.00 | 0.00 |
| fake_front | CleanCoop oracle | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| fake_front | Raw fusion | 2.175% | 100.00% | 87/4000 | NA | 0.00 | 0.00 |
| fake_front | TrustCalib only | 2.175% | 100.00% | 87/4000 | NA | 0.00 | 0.00 |
| fake_front | w/o MissingRecovery | 0.425% | 56.50% | 17/4000 | 100.00% | 0.00 | 0.00 |
| fake_front | w/o BoxGuard | 0.400% | 56.50% | 16/4000 | 94.75% | 0.00 | 0.00 |
| fake_front | w/o smoothing | 0.425% | 56.50% | 17/4000 | 100.00% | 0.00 | 0.00 |
| fake_front | w/o noisy recovery | 0.425% | 56.50% | 17/4000 | 100.00% | 0.00 | 0.00 |
| fake_front | Final | 0.425% | 56.50% | 17/4000 | 100.00% | 0.00 | 0.00 |
| noise+fake_front | EgoOnly | 1.850% | 53.75% | 74/4000 | NA | 0.00 | 0.00 |
| noise+fake_front | CleanCoop oracle | 0.425% | 56.50% | 17/4000 | NA | 0.00 | 0.00 |
| noise+fake_front | Raw fusion | 1.925% | 100.00% | 77/4000 | NA | 0.00 | 0.00 |
| noise+fake_front | TrustCalib only | 2.925% | 91.50% | 117/4000 | NA | 0.00 | 0.00 |
| noise+fake_front | w/o MissingRecovery | 0.700% | 57.00% | 28/4000 | 99.37% | 0.00 | 13.04 |
| noise+fake_front | w/o BoxGuard | 2.425% | 90.75% | 97/4000 | NA | 4.29 | 13.07 |
| noise+fake_front | w/o smoothing | 0.450% | 57.25% | 18/4000 | 99.37% | 6.69 | 0.00 |
| noise+fake_front | w/o noisy recovery | 0.700% | 57.00% | 28/4000 | 99.37% | 0.05 | 13.04 |
| noise+fake_front | Final | 0.425% | 57.00% | 17/4000 | 99.37% | 4.30 | 13.04 |

Interpretation:

- `Raw fusion` and `TrustCalib only` are strong non-safety-aware baselines: they expose the failure modes that calibration alone cannot solve.
- `w/o MissingRecovery` fails on dropout, `w/o BoxGuard` fails on fake/noise+fake, and `w/o smoothing` is slightly worse on compound noise.
- `Final` recovers clean-level WPC on all four evaluated modes under reliable GT-derived support.

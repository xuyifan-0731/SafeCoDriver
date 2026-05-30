| setting | EgoOnly WPC | RealPrimaryTrustCalib WPC | RealMultiEvidenceGuard WPC | avg missing recovered | missing precision | missing recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| min2-baseline-20x20 | 1.850% | 1.725% | 1.625% | 1.20 | 98.75% | 9.12% |
| min1-unconstrained-20x20 | 1.850% | 1.725% | 1.550% | 4.08 | 80.50% | 25.25% |
| pathrisk-thr0-20x20 | 1.850% | 1.725% | 1.550% | 1.21 | 98.76% | 9.17% |
| pathrisk-thr5-20x20 | 1.850% | 1.725% | 1.550% | 1.22 | 98.77% | 9.27% |
| pathrisk+temporal-20x20 | 1.850% | 1.725% | 1.550% | 1.22 | 98.77% | 9.23% |
| pathrisk-thr0-v1-20x20 | 1.850% | 1.725% | 1.550% | 1.21 | 98.76% | 9.17% |
| min2-baseline-full | 1.888% | 1.580% | 1.574% | 1.10 | 92.06% | 9.50% |
| pathrisk-thr0-full | 1.888% | 1.580% | 1.563% | 1.10 | 92.08% | 9.52% |

Diagnostics for `pathrisk-thr0-20x20`:

| diagnostic | value |
| --- | ---: |
| recovered by strict min2 evidence | 480 |
| recovered by path-risk single-source rule | 3 |
| path-risk single-source TP / FP | 3 / 0 |
| skipped insufficient-evidence candidates | 1217 |
| frames better / worse / tied vs EgoOnly | 14 / 6 / 380 |

Interpretation:

- Unconstrained single-source recovery improves WPC but admits many extra objects and reduces precision to `80.50%`.
- Path-risk-aware single-source recovery matches the unconstrained WPC (`1.550%`) while keeping precision near the strict min2 baseline (`98.76%`).
- The effective policy is conservative: keep strict two-source evidence as the default, and admit a one-source missing candidate only when its oriented-box path margin is non-positive.
- The optional temporal path-risk variant recovered additional TP objects but did not further reduce WPC on this 20x20 subset. It remains useful as a diagnostic mechanism rather than the current recommended policy.

Outputs:

```text
results/realmultisource_20x20_pathrisk_min2_thr0/summary.csv
results/realmultisource_20x20_pathrisk_min2_thr0/frame_diagnostics.csv
results/realmultisource_20x20_pathrisk_min2_thr0/cluster_records.csv
results/realmultisource_full_fast_pathrisk_min2_thr0/summary.csv
results/realmultisource_v1_20x20_pathrisk_min2_thr0/summary.csv
```

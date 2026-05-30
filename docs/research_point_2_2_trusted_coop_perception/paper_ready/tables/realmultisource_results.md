| setting | method | WPC | Warn | avg_missing_recovered | missing_precision | missing_recall | avg_sources_available |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20x20-min2-self-filtered | EgoOnly | 1.850% | 40.25% | 0.00 | 0.00% | 0.00% | 4.00 |
| 20x20-min2-self-filtered | CleanCoop oracle | 0.425% | 46.50% | 0.00 | 0.00% | 0.00% | 4.00 |
| 20x20-min2-self-filtered | RealPrimaryRaw | 1.725% | 41.00% | 0.00 | 0.00% | 0.00% | 4.00 |
| 20x20-min2-self-filtered | RealPrimaryTrustCalib | 1.725% | 41.00% | 0.00 | 0.00% | 0.00% | 4.00 |
| 20x20-min2-self-filtered | RealMultiEvidenceGuard | 1.625% | 41.75% | 1.20 | 98.75% | 9.12% | 4.00 |

Diagnostics:

| diagnostic | value |
| --- | ---: |
| frames | 400 |
| frames where RealMultiEvidenceGuard worse than EgoOnly | 6 |
| frames where RealMultiEvidenceGuard better than EgoOnly | 12 |
| frames tied with EgoOnly | 382 |
| missing-recovery TP / FP | 474 / 6 |
| primary-object records with negative box margin | 11 / 2162 |
| missing-recovery records with negative box margin | 3 / 480 |

Interpretation:

- After target-ego duplicate filtering, real multi-source evidence becomes slightly beneficial over EgoOnly (`1.625%` vs `1.850%`) and over the real primary sender alone (`1.725%`), but remains far from the CleanCoop oracle (`0.425%`).
- The current real multi-source bottleneck is low recall, not precision: missing recovery precision is high (`98.75%`), but recall is only `9.12%`.
- This supports a disciplined claim: calibration and evidence-gated recovery work, but deployable real multi-source cooperative perception still needs temporal tracking and path-risk-aware object admission to approach oracle-level safety.

Outputs:

```text
results/realmultisource_20x20_fast_min2_self_filtered_diag/summary.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/frame_diagnostics.csv
results/realmultisource_20x20_fast_min2_self_filtered_diag/cluster_records.csv
```

Invalidated earlier diagnostic:

```text
results/realmultisource_20x20_fast_min2/
```

The earlier `2.875%` WPC result included source-label copies of the target ego vehicle as cooperative obstacles. It is retained only as an audit trail for input-model debugging and should not be used as a paper result.

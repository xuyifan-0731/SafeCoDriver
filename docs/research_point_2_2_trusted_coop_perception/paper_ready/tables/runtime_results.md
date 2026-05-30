| benchmark | command output | frames or mode-frames | wall time | normalized time | max RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| synthetic final, 20x20, clean + noise+fake_front | `results/runtime_probe_synthetic_final_20x20/summary.csv` | 800 mode-frames | 41.46 s | 51.8 ms/mode-frame | 472716 KB |
| real single-sender, 20x20, self-filtered | `results/runtime_probe_real_single_20x20/summary.csv` | 400 frames | 6.87 s | 17.2 ms/frame | 469976 KB |
| real multi-source, 20x20 min2, self-filtered | `results/runtime_probe_real_multisource_20x20/summary.csv` | 400 frames | 11.19 s | 28.0 ms/frame | 471464 KB |

Notes:

- These are end-to-end Python script timings, including dataset loading, calibration, all reported baseline methods inside each script, and CSV writing. They should be reported as reproducibility/runtime-overhead measurements, not optimized deployment latency.
- The synthetic benchmark runs two anomaly modes and reports normalized time per mode-frame.
- All timings were measured in conda environment `Android-Lab` on the current workstation with `torch.set_num_threads(1)`.

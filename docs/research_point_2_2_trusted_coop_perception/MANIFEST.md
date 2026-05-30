# Research Point 2.2 Upload Manifest

This folder is the curated upload package for:

```text
空间失配场景下考虑异常信息的安全可信协同感知优化
```

## Included

### Method And Integration Docs

```text
README.md
01_method_design.md
02_code_integration_plan.md
03_experiment_plan.md
api_blueprint.py
```

These files describe the original mechanism design, SafeCoDriver integration plan, experiment plan, and API blueprint.

### Prototype Code

```text
prototype/anomaly_injection.py
prototype/real_coop.py
prototype/run_deepaccident_mixed_pilot.py
prototype/run_deepaccident_multipeer_pilot.py
prototype/run_deepaccident_realcoop_alignment_audit.py
prototype/run_deepaccident_realcoop_pilot.py
prototype/run_deepaccident_realmultisource_pilot.py
prototype/run_deepaccident_shift_pilot.py
prototype/trust_calib.py
```

Important implementation details:

- `real_coop.py` implements calibrated source-label to ego-label-frame alignment.
- Real cooperative input now filters target-ego duplicate objects within 2.5 m of ego origin.
- `run_deepaccident_realmultisource_pilot.py` can emit frame-level and object-level diagnostics via `--write-diagnostics`.

### Paper-Ready Tables And Status

```text
paper_ready/A_CLASS_EXPERIMENT_FLOW_AND_STATUS_2026-05-30.md
paper_ready/tables/main_synthetic_results.md
paper_ready/tables/ablation_results_20x20_fast.md
paper_ready/tables/seed_robustness.md
paper_ready/tables/support_quality_results.md
paper_ready/tables/realcoop_results.md
paper_ready/tables/realmultisource_results.md
paper_ready/tables/realmultisource_pathrisk_results.md
paper_ready/tables/statistical_intervals.md
paper_ready/tables/paired_bootstrap_real_multisource.md
paper_ready/tables/runtime_results.md
paper_ready/realcoop_alignment_summary_self_filtered.json
```

### Lightweight Result Summaries

```text
results_summaries/
```

This contains selected `summary.csv`, `metadata.json`, `summary.json`, and small diagnostic CSV files for the main paper claims. It intentionally excludes large intermediate cluster records except for small curated diagnostics.

### Research Notes

```text
research_notes/
```

This contains literature triage, closest-prior-work analysis, method constraints, experiment notes, novelty check, paper positioning, and automation status. PDF files, full extracted PDF text, and raw large downloads are excluded.

## Excluded

The following are intentionally not uploaded:

```text
DeepAccident dataset
model checkpoints
conda environments
large cluster_records.csv files
full cached results directory
downloaded PDF papers
PDF extracted text dumps
raw search cache dumps not needed for review
```

## Reproduction Environment

Expected environment:

```bash
conda activate coop-safety
```

The prototype experiments were developed and rerun under:

```bash
conda activate Android-Lab
```

Dataset and checkpoint assumptions match the repository-level `AGENTS.md`:

```text
DeepAccident val: checkpoint scenario split, 22 validation scenarios used by current checkpoint metadata.
SafeCoDriver model: models/collision_net_best.pt
```

## Core Commands

Synthetic final method:

```bash
python docs/research_point_2_2_trusted_coop_perception/prototype/run_deepaccident_multipeer_pilot.py \
  --support-modes clean,shift \
  --enable-time-calib \
  --enable-missing-recovery \
  --missing-recovery-primary-actions accept,correct,time_correct,downweight,quarantine \
  --enable-box-margin-guard \
  --enable-peer-consensus-smoothing
```

Real single-sender corrected pilot:

```bash
python docs/research_point_2_2_trusted_coop_perception/prototype/run_deepaccident_realcoop_pilot.py \
  --max-scenarios 20 \
  --max-frames-per-scenario 20 \
  --disable-v1
```

Real multi-source corrected diagnostic:

```bash
python docs/research_point_2_2_trusted_coop_perception/prototype/run_deepaccident_realmultisource_pilot.py \
  --max-scenarios 20 \
  --max-frames-per-scenario 20 \
  --disable-v1 \
  --min-peer-support 2 \
  --missing-min-peer-support 2 \
  --enable-box-margin-guard \
  --write-diagnostics
```

Real multi-source path-risk admission:

```bash
python docs/research_point_2_2_trusted_coop_perception/prototype/run_deepaccident_realmultisource_pilot.py \
  --max-scenarios 20 \
  --max-frames-per-scenario 20 \
  --disable-v1 \
  --min-peer-support 2 \
  --missing-min-peer-support 2 \
  --enable-box-margin-guard \
  --enable-missing-path-risk-single-support \
  --missing-path-risk-box-margin-thr 0.0 \
  --write-diagnostics
```

## Claim Discipline

Safe main claim:

```text
Under reliable multi-peer evidence, the proposed information-usability framework
restores clean-level waypoint safety across spatial shift, temporal staleness,
dropout, fake-object injection, and noise+fake compound anomalies.
```

Safe real-data claim:

```text
Real DeepAccident cooperative labels can be aligned into the ego frame with
near-zero median residual. After filtering target-ego duplicates, real
multi-source evidence is directionally beneficial but still recall-limited and
far from CleanCoop oracle. Path-risk-aware one-source admission improves the
real 20x20 multi-source WPC from 1.625% to 1.550% while preserving high
missing-recovery precision.
```

Avoid claiming full real multi-vehicle validation or production-ready V2X perception.

# Restore Local Artifacts

The GitHub upload keeps the reproducible source, summaries, lightweight assets,
and the archived V2Xverse patch. It does not upload large datasets, CARLA
installations, third-party working trees, non-final model weights, or generated
logs. The final reproducible checkpoint `models/collision_net_best.pt` is the
one deliberate model-weight exception for the 260728 GitHub backup.

No local files were deleted by the upload. The paths below remain available on
this machine unless they are removed manually.

## Ignored Local Paths

The repository ignores these categories:

- `data/`: DAIR-V2X, DeepAccident, CARLA-derived risk labels, and other local
  datasets.
- `third_party/`: OpenCDA, OpenCOOD, V2Xverse, CARLA 0.9.15, CARLA 0.9.10.1,
  downloaded checkpoints, build products, and V2Xverse datasets.
- `models/`: trained model weights are ignored except
  `models/collision_net_best.pt`, which is tracked for the final reproduction
  path.
- generated logs and outputs: `*.log`, `nohup.out`, `experiments/results/`,
  most `results/`, and temporary/cache files.
- large archives and point clouds such as `*.zip`, `*.tar.gz`, and large
  CARLA/V2Xverse/DeepAccident files.

## Restore V2Xverse Local Code Changes

`third_party/V2Xverse` is an independent third-party Git repository. Its local
changes were archived under `third_party_patches/V2Xverse/` instead of being
pushed to the upstream V2Xverse repository.

From the SafeCoDriver repository root:

```bash
mkdir -p third_party
git clone https://github.com/CollaborativePerception/V2Xverse.git third_party/V2Xverse
git -C third_party/V2Xverse apply ../../third_party_patches/V2Xverse/tracked_changes.patch
cp -a third_party_patches/V2Xverse/untracked/. third_party/V2Xverse/
```

This restores:

- modifications to V2Xverse tracked files;
- local safety-evaluation scripts/configs;
- the uploaded text evaluation outputs under `eval_output/` and
  `eval_output_safety/`.

It does not restore the V2Xverse dataset archives, model checkpoints, caches,
or build directories.

## Restore Third-Party Repositories

Recreate ignored third-party code checkouts as needed:

```bash
mkdir -p third_party
git clone https://github.com/ucla-mobility/OpenCDA.git third_party/OpenCDA
git clone https://github.com/DerrickXuNu/OpenCOOD.git third_party/OpenCOOD
git clone https://github.com/CollaborativePerception/V2Xverse.git third_party/V2Xverse
```

Then apply the V2Xverse local changes using the commands above.

## Restore CARLA

The local workspace contains ignored CARLA installs:

```text
third_party/carla/
third_party/carla_0910/
```

They correspond to local CARLA archives such as:

```text
third_party/carla/CARLA_0.9.15.tar.gz
third_party/carla_0910/CARLA_0.9.10.1.tar.gz
```

To restore on a new machine, download the matching CARLA Linux releases from
the official CARLA release page or copy the existing local archives, then
extract them back to those paths. Keep HD maps and Unreal asset files inside
the ignored CARLA directories; they are too large for Git.

## Restore Datasets

Restore datasets by copying or downloading them into the ignored `data/`
layout:

```text
data/DAIR-V2X/
data/DAIR-V2X-C-Example/
data/DAIR-V2X-C-Full/
data/DeepAccident/
data/carla_risk_labels/
data/risk_labels/
```

Large local archives seen in the workspace include:

```text
data/DeepAccident/val_part1.zip
data/DeepAccident/val_part2.zip
data/DAIR-V2X-C-Example.zip
data/DAIR-V2X-C-Full/cooperative-vehicle-infrastructure.zip
```

For V2Xverse, restore the dataset under:

```text
third_party/V2Xverse/dataset/
```

Those V2Xverse route archives are often hundreds of MB to several GB each and
are intentionally not mirrored in Git.

## Restore Models And Generated Results

If a workflow expects non-final trained weights under `models/` or third-party
checkpoint directories, restore them from local storage or regenerate them by
rerunning the corresponding training/evaluation scripts. The final Research
Point 3 reproduction path should already have `models/collision_net_best.pt` in
Git.

Generated logs do not need to be restored. They are intentionally excluded:

- `*.log`
- `nohup.out`
- V2Xverse build/cache directories
- full experiment result folders not explicitly whitelisted in `.gitignore`

The repository includes the lightweight summaries and patch artifacts needed to
reconstruct the source state; large runtime products should stay local.

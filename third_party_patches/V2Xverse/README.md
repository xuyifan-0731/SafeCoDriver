# V2Xverse Local Changes

This directory archives local changes made inside the ignored nested repository
`third_party/V2Xverse`.

The nested repository's remote is the upstream project:

```text
https://github.com/CollaborativePerception/V2Xverse.git
```

These changes are stored here so they are uploaded with the SafeCoDriver
repository without pushing directly to that third-party upstream.

## Contents

- `tracked_changes.patch`: patch for modified files already tracked by
  V2Xverse.
- `untracked/`: local files that were untracked inside V2Xverse, preserving
  their original relative paths.

## Restore Into `third_party/V2Xverse`

From the SafeCoDriver repository root:

```bash
git -C third_party/V2Xverse apply ../../third_party_patches/V2Xverse/tracked_changes.patch
cp -a third_party_patches/V2Xverse/untracked/. third_party/V2Xverse/
```

The V2Xverse dataset zips and other large third-party artifacts are not archived
here.

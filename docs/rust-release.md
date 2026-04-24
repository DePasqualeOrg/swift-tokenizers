# Rust release workflow (maintainer guide)

This document explains how the `rust-release.yml` workflow publishes new Rust XCFramework artifacts and how maintainers cut releases. It is scoped to the Rust binary; the Swift package release process is separate.

## Overview

`rust-release.yml` is a `workflow_dispatch` workflow split into two jobs:

- **`build-validate`**: verifies the dispatched commit matches `expected_commit`, builds the XCFramework, runs `swift test --traits Rust` against the freshly built artifact, computes a canonical source hash, writes a release manifest JSON, and uploads all four files (archive, checksum, manifest, release notes) as workflow artifacts. Runs under `contents: read`.
- **`publish`**: downloads the workflow artifacts and creates the GitHub release via `gh release create --target "$GITHUB_SHA" ...`. Runs under `contents: write` and is gated by a GitHub Environment (`rust-release`) with the maintainer as a required reviewer.

The split puts the approval prompt exactly at the write boundary. A failed build or a mismatched `expected_commit` fails the workflow before the publish job is ever queued for approval.

## One-time setup

### GitHub Environment

Create a repository Environment named `rust-release` under **Settings → Environments → New environment**:

1. Add the maintainer as a required reviewer.
2. Confirm "Prevent self-review" is **disabled** (the maintainer is their own reviewer).
3. Save.

The `publish` job references `environment: rust-release`. Without this configuration the publish job fails when it tries to enter the environment.

## Cutting a release

Run `scripts/rust/release/cut-release.sh <version>` from the branch you want to publish. The script:

1. Validates the working tree is clean and the branch is pushed to origin.
2. Records local `HEAD`.
3. Dispatches `rust-release.yml` against the current branch with `version` and `expected_commit=<local HEAD>`.
4. Waits for the run to finish.
5. Verifies `gh run view --json headSha` equals the expected commit.
6. Prints the release URL on success.

Automated writing of `rust/Pin.json` is not yet wired up. Until the Pin.json PR lands, the script stops after publishing and prints a note explaining that the pin bump must be done manually (or by extending the script in a follow-up).

### Bootstrap 0.3.2

The first release under the new workflow is `tokenizers-rust-0.3.2`, cut from `main` to produce the baseline manifest. From a clean `main` working tree:

```
scripts/rust/release/cut-release.sh 0.3.2
```

The workflow needs `expected_commit` to match the dispatched ref's commit. Running from `main` after a fresh `git pull origin main` satisfies that.

## Trust boundary

`rust-release.yml` has `contents: write` (on the `publish` job) and runs the workflow file from whatever ref is dispatched. A branch that modifies the workflow could publish bogus assets if a maintainer dispatches the release against it.

- Do not dispatch `rust-release.yml` against an external contributor's PR branch directly. Merge or cherry-pick the reviewed Rust source changes onto a maintainer-owned branch (or `main`) first, then run `cut-release.sh` there.
- Treat the release workflow and `cut-release.sh` as maintainer-only tools. External contributors cannot trigger `workflow_dispatch` – only repository write access can – but this discipline protects against social-engineering dispatches.

## What this PR does not yet cover

The following pieces complete the workflow and land in a follow-up PR:

- `rust/Pin.json` sidecar file seeded from the `0.3.2` manifest.
- `Package.swift` reading the URL and checksum from `rust/Pin.json`.
- CI drift-guard job that fails when the Rust source tree drifts from `Pin.json.source_hash_sha256`.
- Pre-release pin policy (reject `-rc` versions on PRs targeting `main` unless explicitly allowed).
- Completion of `cut-release.sh` to download the manifest asset and commit the `Pin.json` bump.

#!/usr/bin/env bash
# Emit a canonical SHA-256 hash of the Rust source tree and related build
# inputs, as a single hex string to stdout.
#
# Used by the release workflow to pin the published XCFramework manifest to
# the exact source state that produced it, and by the CI drift guard to
# detect when the pinned binary no longer matches the source tree.
#
# Inputs: all git-tracked files under the paths in ALLOWLIST below.
# Output: plain hex SHA-256 to stdout.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ALLOWLIST=(
    rust/swift-tokenizers-rust
    rust-toolchain.toml
    scripts/rust/build
    .github/workflows/rust-release.yml
)

(
    git ls-files -z "${ALLOWLIST[@]}" \
        | LC_ALL=C sort -z \
        | while IFS= read -r -d '' path; do
            content_hash=$(shasum -a 256 "$path" | awk '{print $1}')
            printf '%s %s\0' "$content_hash" "$path"
        done
) | shasum -a 256 | awk '{print $1}'

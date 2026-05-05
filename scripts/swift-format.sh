#!/usr/bin/env bash
# Lint or format the Swift sources in this package. The path list is centralised
# here so the pre-commit hook, the CI lint job, and any local invocation all
# operate on exactly the same set of files. Build artifacts under `rust/target/`
# are excluded by virtue of not being listed. The UniFFI-generated wrapper at
# `Sources/TokenizersFFI/Generated/TokenizersFFI.swift` is recursively in scope
# but exempted via the `// swift-format-ignore-file` marker prepended by
# `scripts/rust/regenerate-wrapper.sh` and `scripts/rust/check-wrapper-drift.sh`.
#
# Usage:
#   scripts/swift-format.sh           # lint, fail on any diagnostic
#   scripts/swift-format.sh --fix     # format in place

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PATHS=(
    Package.swift
    Sources
    Tests
)

mode="${1:-lint}"
case "${mode}" in
    lint)
        cd "${REPO_ROOT}"
        exec swift format lint --strict --recursive "${PATHS[@]}"
        ;;
    --fix|fix)
        cd "${REPO_ROOT}"
        exec swift format -i --recursive "${PATHS[@]}"
        ;;
    *)
        echo "Usage: $(basename "$0") [lint|--fix]" >&2
        exit 2
        ;;
esac

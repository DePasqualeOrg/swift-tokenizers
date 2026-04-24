#!/usr/bin/env bash
# Build documentation for all library targets under both the Swift and Rust
# traits. Treats doc warnings (broken links, missing symbols, malformed
# blocks) as errors so regressions are caught in CI.
#
# The Swift and Rust backends are mutually exclusive, so docs are generated
# in two passes. Each pass covers the trait-independent targets (Tokenizers
# facade and TokenizersCore) plus the backend target for that trait.
#
# Requires TOKENIZERS_ENABLE_DOCS=1 so Package.swift resolves the
# swift-docc-plugin dependency. Without this guard, end users resolving the
# package would pull in the plugin unnecessarily.

set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_ENABLE_DOCS=1

run_pass() {
    local trait="$1"
    local backend_target="$2"

    echo
    echo "=== Generating documentation with --traits ${trait} ==="

    swift package clean
    swift package --traits "${trait}" generate-documentation \
        --target Tokenizers \
        --target TokenizersCore \
        --target "${backend_target}" \
        --warnings-as-errors
}

run_pass Swift TokenizersSwiftBackend
run_pass Rust TokenizersRustBackend

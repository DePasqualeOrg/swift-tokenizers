#!/usr/bin/env bash
# Build documentation for all library targets. Treats doc warnings (broken
# links, missing symbols, malformed blocks) as errors so regressions are caught
# in CI.
#
# Requires TOKENIZERS_ENABLE_DOCS=1 so Package.swift resolves the
# swift-docc-plugin dependency. Without this guard, end users resolving the
# package would pull in the plugin unnecessarily.

set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_ENABLE_DOCS=1

echo
echo "=== Generating documentation ==="

swift package clean
swift package generate-documentation \
    --target Tokenizers \
    --warnings-as-errors

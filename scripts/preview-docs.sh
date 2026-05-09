#!/usr/bin/env bash
# Preview documentation with live reload at http://localhost:8080/documentation/tokenizers.
#
# Requires TOKENIZERS_ENABLE_DOCS=1 so Package.swift resolves the
# swift-docc-plugin dependency. The script sets it for you.
#
# docc preview watches the `.docc` catalog for changes to Markdown articles,
# so editing the landing page reloads automatically. Source (Swift) changes do
# NOT trigger re-extraction of symbol graphs — rerun the script after editing
# doc comments in Swift source.

set -euo pipefail
cd "$(dirname "$0")/.."

export TOKENIZERS_ENABLE_DOCS=1

# Use a locally assembled Rust artifactbundle if one exists, so the build
# doesn't fail on a published artifact that lags the Swift source.
local_bundle="rust/target/artifactbundle/TokenizersRust.artifactbundle"
if [[ -d "${local_bundle}" && -z "${TOKENIZERS_RUST_LOCAL_ARTIFACTBUNDLE_PATH:-}" ]]; then
    export TOKENIZERS_RUST_LOCAL_ARTIFACTBUNDLE_PATH="${local_bundle}"
fi

swift package --disable-sandbox preview-documentation --target Tokenizers

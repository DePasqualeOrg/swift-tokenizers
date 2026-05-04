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

# Use the locally built Rust XCFramework if it exists, so the build doesn't fail
# on a published artifact that lags the Swift source.
if [[ -d "Binaries/TokenizersRust.xcframework" && -z "${TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH:-}" ]]; then
    export TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH="Binaries/TokenizersRust.xcframework"
fi

swift package --disable-sandbox preview-documentation --target Tokenizers

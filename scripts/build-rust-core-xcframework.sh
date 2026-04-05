#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/rust/swift-tokenizers-rust-core"
HEADERS_DIR="${CRATE_DIR}/include"
OUTPUT_DIR="${REPO_ROOT}/Binaries/TokenizersRustCore.xcframework"
INTERMEDIATES_DIR="${CRATE_DIR}/target/xcframework-intermediates"

export CARGO_TARGET_DIR="${CRATE_DIR}/target"

TARGETS=(
  aarch64-apple-darwin
  x86_64-apple-darwin
  aarch64-apple-ios
  aarch64-apple-ios-sim
  x86_64-apple-ios
)

for target in "${TARGETS[@]}"; do
  echo "Building for ${target}..."
  cargo build \
    --manifest-path "${CRATE_DIR}/Cargo.toml" \
    --release \
    --target "${target}"
done

rm -rf "${OUTPUT_DIR}"
rm -rf "${INTERMEDIATES_DIR}"
mkdir -p "${INTERMEDIATES_DIR}"

MACOS_UNIVERSAL_LIB="${INTERMEDIATES_DIR}/libtokenizers_rust_core-macos.a"
IOS_SIM_UNIVERSAL_LIB="${INTERMEDIATES_DIR}/libtokenizers_rust_core-ios-simulator.a"

lipo -create \
  "${CRATE_DIR}/target/aarch64-apple-darwin/release/libtokenizers_rust_core.a" \
  "${CRATE_DIR}/target/x86_64-apple-darwin/release/libtokenizers_rust_core.a" \
  -output "${MACOS_UNIVERSAL_LIB}"

lipo -create \
  "${CRATE_DIR}/target/aarch64-apple-ios-sim/release/libtokenizers_rust_core.a" \
  "${CRATE_DIR}/target/x86_64-apple-ios/release/libtokenizers_rust_core.a" \
  -output "${IOS_SIM_UNIVERSAL_LIB}"

xcodebuild -create-xcframework \
  -library "${MACOS_UNIVERSAL_LIB}" \
  -headers "${HEADERS_DIR}" \
  -library "${CRATE_DIR}/target/aarch64-apple-ios/release/libtokenizers_rust_core.a" \
  -headers "${HEADERS_DIR}" \
  -library "${IOS_SIM_UNIVERSAL_LIB}" \
  -headers "${HEADERS_DIR}" \
  -output "${OUTPUT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/rust/swift-tokenizers-rust"
HEADERS_DIR="${CRATE_DIR}/include"
OUTPUT_DIR="${REPO_ROOT}/Binaries/TokenizersRust.xcframework"
INTERMEDIATES_DIR="${CRATE_DIR}/target/xcframework-intermediates"
LOCKFILE_PATH="${CRATE_DIR}/Cargo.lock"
TOOLCHAIN_FILE="${REPO_ROOT}/rust-toolchain.toml"

export CARGO_TARGET_DIR="${CRATE_DIR}/target"

if [[ ! -f "${LOCKFILE_PATH}" ]]; then
  echo "Missing ${LOCKFILE_PATH}. Commit the lockfile before building release artifacts." >&2
  exit 1
fi

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Missing ${TOOLCHAIN_FILE}. Pin the Rust toolchain before building release artifacts." >&2
  exit 1
fi

TOOLCHAIN="$(python3 - <<'PY' "${TOOLCHAIN_FILE}"
import pathlib
import sys
import tomllib

toolchain_file = pathlib.Path(sys.argv[1])
data = tomllib.loads(toolchain_file.read_text())
print(data["toolchain"]["channel"])
PY
)"

TARGETS=(
  aarch64-apple-darwin
  x86_64-apple-darwin
  aarch64-apple-ios
  aarch64-apple-ios-sim
  x86_64-apple-ios
)

rustup toolchain install "${TOOLCHAIN}" --profile minimal
rustup target add --toolchain "${TOOLCHAIN}" "${TARGETS[@]}"

for target in "${TARGETS[@]}"; do
  echo "Building for ${target}..."
  cargo build \
    --manifest-path "${CRATE_DIR}/Cargo.toml" \
    --locked \
    --release \
    --target "${target}"
done

rm -rf "${OUTPUT_DIR}"
rm -rf "${INTERMEDIATES_DIR}"
mkdir -p "${INTERMEDIATES_DIR}"

MACOS_UNIVERSAL_LIB="${INTERMEDIATES_DIR}/libtokenizers_rust-macos.a"
IOS_SIM_UNIVERSAL_LIB="${INTERMEDIATES_DIR}/libtokenizers_rust-ios-simulator.a"

lipo -create \
  "${CRATE_DIR}/target/aarch64-apple-darwin/release/libtokenizers_rust.a" \
  "${CRATE_DIR}/target/x86_64-apple-darwin/release/libtokenizers_rust.a" \
  -output "${MACOS_UNIVERSAL_LIB}"

lipo -create \
  "${CRATE_DIR}/target/aarch64-apple-ios-sim/release/libtokenizers_rust.a" \
  "${CRATE_DIR}/target/x86_64-apple-ios/release/libtokenizers_rust.a" \
  -output "${IOS_SIM_UNIVERSAL_LIB}"

xcodebuild -create-xcframework \
  -library "${MACOS_UNIVERSAL_LIB}" \
  -headers "${HEADERS_DIR}" \
  -library "${CRATE_DIR}/target/aarch64-apple-ios/release/libtokenizers_rust.a" \
  -headers "${HEADERS_DIR}" \
  -library "${IOS_SIM_UNIVERSAL_LIB}" \
  -headers "${HEADERS_DIR}" \
  -output "${OUTPUT_DIR}"

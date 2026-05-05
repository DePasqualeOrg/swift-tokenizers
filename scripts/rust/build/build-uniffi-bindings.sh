#!/usr/bin/env bash
set -euo pipefail

# Builds the UniFFI Swift wrapper, generated C header, and modulemap into
# `rust/target/uniffi-bindings/`.
#
# This is the cheap subset of `build-rust-xcframework.sh`: it builds only the
# host static library plus the `uniffi-bindgen-swift` binary, not the full
# per-target matrix or the xcframework assembly. UniFFI's library-mode bindgen
# reads metadata from one static library, and the metadata is identical across
# the per-target slices we ship, so a single host-target build is enough to
# regenerate the wrapper.
#
# Callers:
# - `scripts/rust/regenerate-wrapper.sh` (developer command that copies the
#    generated wrapper over the committed copy).
# - `scripts/rust/check-wrapper-drift.sh` (CI drift guard).
# - `scripts/rust/build/build-rust-xcframework.sh` (full release build, after
#    the per-target `cargo build` invocations).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CRATE_DIR="${REPO_ROOT}/rust"
UNIFFI_BINDINGS_DIR="${CRATE_DIR}/target/uniffi-bindings"
LOCKFILE_PATH="${CRATE_DIR}/Cargo.lock"
TOOLCHAIN_FILE="${REPO_ROOT}/rust-toolchain.toml"

export CARGO_TARGET_DIR="${CRATE_DIR}/target"

if [[ ! -f "${LOCKFILE_PATH}" ]]; then
  echo "Missing ${LOCKFILE_PATH}. Commit the lockfile before building bindgen artifacts." >&2
  exit 1
fi

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Missing ${TOOLCHAIN_FILE}. Pin the Rust toolchain before building bindgen artifacts." >&2
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

# We feed bindgen the host static library. The metadata is identical across
# the per-target builds, so any one slice works; we pick `aarch64-apple-darwin`
# because the macOS CI runners and developer machines we target are all
# Apple Silicon.
HOST_TARGET="aarch64-apple-darwin"

rustup toolchain install "${TOOLCHAIN}" --profile minimal
rustup target add --toolchain "${TOOLCHAIN}" "${HOST_TARGET}"

cargo +"${TOOLCHAIN}" build \
  --manifest-path "${CRATE_DIR}/Cargo.toml" \
  --locked \
  --release \
  --target "${HOST_TARGET}"

cargo +"${TOOLCHAIN}" build \
  --manifest-path "${CRATE_DIR}/Cargo.toml" \
  --locked \
  --release \
  --features uniffi-cli \
  --bin uniffi-bindgen-swift

BINDGEN_BIN="${CRATE_DIR}/target/release/uniffi-bindgen-swift"
HOST_STATIC_LIB="${CRATE_DIR}/target/${HOST_TARGET}/release/libtokenizers_rust.a"

rm -rf "${UNIFFI_BINDINGS_DIR}"
mkdir -p "${UNIFFI_BINDINGS_DIR}"

# UniFFI's library-mode bindgen reads metadata from the static library. The
# Swift wrapper goes alongside the headers in `target/uniffi-bindings/`; this
# script never copies it into `Sources/TokenizersFFI/Generated/` — that's
# `scripts/rust/regenerate-wrapper.sh`'s job. See
# `docs/uniffi-migration.md` § "Generated-wrapper drift integrity".
#
# UniFFI v0.31.x discovers `uniffi.toml` via `cargo metadata`, which must run
# from a directory inside the crate's manifest tree, so we invoke bindgen with
# `cd "${CRATE_DIR}"`.
(
  cd "${CRATE_DIR}"
  "${BINDGEN_BIN}" \
    --swift-sources \
    "${HOST_STATIC_LIB}" \
    "${UNIFFI_BINDINGS_DIR}"

  # The TokenizersRust XCFramework slices ship `Headers/` + `libtokenizers_rust.a`
  # (not a `.framework` bundle), so the modulemap must be a plain `module`, not
  # `framework module`. We omit `--xcframework`, which is the flag that flips
  # UniFFI's bindgen template to emit the framework form.
  "${BINDGEN_BIN}" \
    --headers --modulemap \
    --module-name TokenizersRust \
    --modulemap-filename module.modulemap \
    "${HOST_STATIC_LIB}" \
    "${UNIFFI_BINDINGS_DIR}"
)

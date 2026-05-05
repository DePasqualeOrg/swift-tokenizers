#!/usr/bin/env bash
set -euo pipefail

# Verifies that the committed UniFFI Swift wrapper at
# `Sources/TokenizersFFI/Generated/TokenizersFFI.swift` matches what the
# bindgen would emit for the current Rust source.
#
# Runs the bindgen-only build (host static library + bindgen binary, no
# per-target matrix or xcframework assembly), prepends the same
# `// swift-format-ignore-file` marker that `regenerate-wrapper.sh` would, and
# diffs the result against the committed wrapper. Nonzero diff → fail with
# instructions to run `scripts/rust/regenerate-wrapper.sh`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

bash "${SCRIPT_DIR}/build/build-uniffi-bindings.sh"

GENERATED_WRAPPER="${REPO_ROOT}/rust/target/uniffi-bindings/TokenizersFFI.swift"
COMMITTED_WRAPPER_DIR="${REPO_ROOT}/Sources/TokenizersFFI/Generated"
COMMITTED_WRAPPER="${COMMITTED_WRAPPER_DIR}/TokenizersFFI.swift"

if [[ ! -f "${GENERATED_WRAPPER}" ]]; then
  echo "Expected generated wrapper at ${GENERATED_WRAPPER} but it does not exist." >&2
  exit 1
fi

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TEMP_DIR}"' EXIT

# TODO: drop this prepend (see the matching block in
# `scripts/rust/regenerate-wrapper.sh`) once a swift-format release including
# apple/swift-format#1197 ships in our stable Swift toolchain — at that point
# a `.swift-format-ignore` file at the repo root replaces it.
{
    echo "// swift-format-ignore-file"
    cat "${GENERATED_WRAPPER}"
} > "${TEMP_DIR}/TokenizersFFI.swift"

if ! git diff --no-index --exit-code "${TEMP_DIR}/TokenizersFFI.swift" "${COMMITTED_WRAPPER}"; then
  cat >&2 <<EOF

Generated UniFFI Swift wrapper drifted from the committed copy. Run

    scripts/rust/regenerate-wrapper.sh

and commit the resulting changes to ${COMMITTED_WRAPPER#${REPO_ROOT}/}.
EOF
  exit 1
fi

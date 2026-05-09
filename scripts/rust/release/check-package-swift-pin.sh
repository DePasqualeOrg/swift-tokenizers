#!/usr/bin/env bash
# Verify that Package.swift's inline Rust artifactbundle pin
# (tokenizersRustArtifactBundleURL + tokenizersRustArtifactBundleChecksum)
# matches rust/Pin.json's artifactbundle_url and checksum fields.
#
# cut-release.sh writes both files in sync when bumping the pinned
# Rust artifact. This check catches drift from manual edits or a buggy
# sync path — consumers of a drifted tag would resolve a checksum that
# doesn't match the URL their Package.swift points at.
#
# Usage: scripts/rust/release/check-package-swift-pin.sh [<Package.swift> <Pin.json>]
#   Defaults to the repo-root files. Callers (e.g. publish-package-release.sh)
#   can pass process-substituted paths to check a specific git ref:
#     check-package-swift-pin.sh \
#       <(git show REF:Package.swift) <(git show REF:rust/Pin.json)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PACKAGE_SWIFT="${1:-${REPO_ROOT}/Package.swift}"
PIN_FILE="${2:-${REPO_ROOT}/rust/Pin.json}"

# Slurp PIN_FILE once so callers can pass process substitutions
# (e.g. `<(git show REF:rust/Pin.json)`); those are single-use streams
# and running jq twice on them would make the second read see EOF.
pin_content="$(cat "${PIN_FILE}")"
pin_url="$(jq -r '.artifactbundle_url' <<<"${pin_content}")"
pin_checksum="$(jq -r '.checksum' <<<"${pin_content}")"

python3 - "${PACKAGE_SWIFT}" "${pin_url}" "${pin_checksum}" <<'PY'
import re
import sys

path, pin_url, pin_checksum = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path, "r", encoding="utf-8") as f:
    text = f.read()
url_match = re.search(r'let tokenizersRustArtifactBundleURL\s*=\s*\n?\s*"([^"]*)"', text)
checksum_match = re.search(r'let tokenizersRustArtifactBundleChecksum\s*=\s*\n?\s*"([^"]*)"', text)
if not url_match or not checksum_match:
    sys.exit("Failed to locate inline Rust artifactbundle pin in Package.swift.")
package_url, package_checksum = url_match.group(1), checksum_match.group(1)
if package_url != pin_url:
    sys.exit(
        f"Package.swift tokenizersRustArtifactBundleURL ({package_url}) "
        f"does not match rust/Pin.json artifactbundle_url ({pin_url})."
    )
if package_checksum != pin_checksum:
    sys.exit(
        f"Package.swift tokenizersRustArtifactBundleChecksum ({package_checksum}) "
        f"does not match rust/Pin.json checksum ({pin_checksum})."
    )
PY

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION="${1:?usage: scripts/package-rust-core-release.sh <version>}"
XCFRAMEWORK_PATH="${REPO_ROOT}/Binaries/TokenizersRustCore.xcframework"
ARTIFACTS_DIR="${REPO_ROOT}/Artifacts"
ARCHIVE_BASENAME="TokenizersRustCore-${VERSION}.xcframework.zip"
ARCHIVE_PATH="${ARTIFACTS_DIR}/${ARCHIVE_BASENAME}"
CHECKSUM_PATH="${ARCHIVE_PATH}.checksum"

bash "${REPO_ROOT}/scripts/build-rust-core-xcframework.sh"

mkdir -p "${ARTIFACTS_DIR}"
rm -f "${ARCHIVE_PATH}" "${CHECKSUM_PATH}"

ditto -c -k --sequesterRsrc --keepParent "${XCFRAMEWORK_PATH}" "${ARCHIVE_PATH}"
swift package compute-checksum "${ARCHIVE_PATH}" | tee "${CHECKSUM_PATH}"

echo
echo "Created ${ARCHIVE_PATH}"
echo "Saved checksum to ${CHECKSUM_PATH}"

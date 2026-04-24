#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VERSION="${1:?usage: scripts/rust/release/publish-rust-release.sh <version> [--ref <ref>]}"
shift || true

REF="main"
WAIT=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref)
      REF="${2:?missing value for --ref}"
      shift 2
      ;;
    --no-wait)
      WAIT=false
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "usage: scripts/rust/release/publish-rust-release.sh <version> [--ref <ref>] [--no-wait]" >&2
      exit 1
      ;;
  esac
done

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Version must be semantic, for example 0.3.0 or 0.3.0-rc.1." >&2
  exit 1
fi

cd "${REPO_ROOT}"

mapfile -t existing_run_ids < <(
  gh run list \
    --workflow rust-release.yml \
    --branch "${REF}" \
    --limit 20 \
    --json databaseId \
    --jq '.[].databaseId'
)

gh workflow run rust-release.yml --ref "${REF}" -f "version=${VERSION}"

echo
echo "Triggered Publish Rust XCFramework for ${VERSION} on ${REF}."

if [[ "${WAIT}" != true ]]; then
  echo "Inspect status with:"
  echo "  gh run list --workflow rust-release.yml --limit 5"
  exit 0
fi

run_id=""
for _ in {1..24}; do
  run_id="$(
    gh run list \
      --workflow rust-release.yml \
      --branch "${REF}" \
      --limit 20 \
      --json databaseId \
      --jq '.[].databaseId' \
      | while IFS= read -r candidate; do
          [[ -z "${candidate}" ]] && continue
          seen=false
          for existing in "${existing_run_ids[@]}"; do
            if [[ "${candidate}" == "${existing}" ]]; then
              seen=true
              break
            fi
          done
          if [[ "${seen}" == false ]]; then
            echo "${candidate}"
            break
          fi
        done
  )"

  if [[ -n "${run_id}" ]]; then
    break
  fi

  sleep 5
done

if [[ -z "${run_id}" ]]; then
  echo "Triggered the workflow, but could not determine the new run id automatically." >&2
  echo "Inspect status with:" >&2
  echo "  gh run list --workflow rust-release.yml --limit 5" >&2
  exit 1
fi

echo "Watching run ${run_id}..."

while true; do
  mapfile -t run_state < <(
    gh run view "${run_id}" \
      --json status,conclusion,url \
      --jq '.status, (.conclusion // ""), .url'
  )
  status="${run_state[0]}"
  conclusion="${run_state[1]}"
  run_url="${run_state[2]}"

  if [[ "${status}" == "completed" ]]; then
    if [[ "${conclusion}" == "success" ]]; then
      release_url="$(gh release view "tokenizers-rust-${VERSION}" --json url --jq '.url')"
      echo "Rust artifact release published successfully."
      echo "Run: ${run_url}"
      echo "Release: ${release_url}"
      exit 0
    fi

    echo "Rust artifact publish failed." >&2
    echo "Run: ${run_url}" >&2
    exit 1
  fi

  sleep 10
done

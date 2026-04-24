#!/usr/bin/env bash
# Dispatch rust-release.yml for the current branch at HEAD, wait for the run
# to finish, and verify the published release was built from the expected
# commit.
#
# Intended to be run on a branch that already contains the Rust source and
# workflow changes you want to publish, with a clean working tree pushed to
# origin. For a normal PR flow this is a Rust-touching PR branch; for the
# bootstrap 0.3.2 release under the new workflow this is main.
#
# A later change will extend this script to also download the manifest asset,
# write rust/Pin.json, and commit the bump on the current branch. For now it
# stops after publishing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VERSION="${1:?usage: scripts/rust/release/cut-release.sh <version> [--no-wait]}"
shift || true

WAIT=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-wait)
      WAIT=false
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "usage: scripts/rust/release/cut-release.sh <version> [--no-wait]" >&2
      exit 1
      ;;
  esac
done

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; then
  echo "Version must be semantic, for example 0.3.0 or 0.3.0-rc.1." >&2
  exit 1
fi

cd "${REPO_ROOT}"

TAG="tokenizers-rust-${VERSION}"
if gh release view "${TAG}" >/dev/null 2>&1; then
  echo "Release ${TAG} already exists. Publish a new semantic version." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean. Commit or stash changes before cutting a release." >&2
  exit 1
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${BRANCH}" == "HEAD" ]]; then
  echo "Detached HEAD. Check out a branch before cutting a release." >&2
  exit 1
fi

EXPECTED_COMMIT="$(git rev-parse HEAD)"

git fetch origin "${BRANCH}" --quiet
REMOTE_COMMIT="$(git rev-parse "origin/${BRANCH}")"
if [[ "${EXPECTED_COMMIT}" != "${REMOTE_COMMIT}" ]]; then
  echo "Local ${BRANCH} (${EXPECTED_COMMIT}) does not match origin/${BRANCH} (${REMOTE_COMMIT})." >&2
  echo "Push your branch before cutting a release so the workflow runs against the expected commit." >&2
  exit 1
fi

mapfile -t existing_run_ids < <(
  gh run list \
    --workflow rust-release.yml \
    --branch "${BRANCH}" \
    --limit 20 \
    --json databaseId \
    --jq '.[].databaseId'
)

gh workflow run rust-release.yml --ref "${BRANCH}" \
  -f "version=${VERSION}" \
  -f "expected_commit=${EXPECTED_COMMIT}"

echo
echo "Dispatched Publish Rust XCFramework for ${VERSION} on ${BRANCH} at ${EXPECTED_COMMIT}."

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
      --branch "${BRANCH}" \
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
      run_head_sha="$(gh run view "${run_id}" --json headSha --jq '.headSha')"
      if [[ "${run_head_sha}" != "${EXPECTED_COMMIT}" ]]; then
        echo "Run ${run_id} completed, but its commit (${run_head_sha}) differs from the expected commit (${EXPECTED_COMMIT})." >&2
        echo "Do not trust this release. Investigate before using the artifact." >&2
        exit 1
      fi

      release_url="$(gh release view "${TAG}" --json url --jq '.url')"
      echo "Rust artifact release published successfully."
      echo "Run: ${run_url}"
      echo "Release: ${release_url}"
      echo
      echo "Note: rust/Pin.json update not yet automated. Extend cut-release.sh in a"
      echo "following PR to download the manifest asset and commit the pin bump."
      exit 0
    fi

    echo "Rust artifact publish failed." >&2
    echo "Run: ${run_url}" >&2
    exit 1
  fi

  sleep 10
done

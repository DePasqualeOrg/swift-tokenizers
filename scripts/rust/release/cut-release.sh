#!/usr/bin/env bash
# Dispatch rust-release.yml for the current branch at HEAD, wait for the run
# to finish, verify the published release was built from the expected
# commit, download the release manifest, write it verbatim to
# rust/Pin.json, and commit the bump on the current branch.
#
# Intended to be run on a branch that already contains the Rust source and
# workflow changes you want to publish, with a clean working tree pushed to
# origin. For a normal PR flow this is a Rust-touching PR branch.

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

      manifest_asset="TokenizersRust-${VERSION}.manifest.json"
      pin_file="${REPO_ROOT}/rust/Pin.json"
      tmp_manifest="$(mktemp)"
      trap 'rm -f "${tmp_manifest}"' EXIT

      echo "Downloading ${manifest_asset} for pin bump..."
      gh release download "${TAG}" \
        --pattern "${manifest_asset}" \
        --output "${tmp_manifest}" \
        --clobber

      manifest_version="$(jq -r '.version' "${tmp_manifest}")"
      manifest_commit="$(jq -r '.git_commit' "${tmp_manifest}")"

      if [[ "${manifest_version}" != "${VERSION}" ]]; then
        echo "Manifest version (${manifest_version}) does not match requested version (${VERSION})." >&2
        exit 1
      fi

      if [[ "${manifest_commit}" != "${EXPECTED_COMMIT}" ]]; then
        echo "Manifest git_commit (${manifest_commit}) does not match expected commit (${EXPECTED_COMMIT})." >&2
        exit 1
      fi

      local_hash="$(bash "${REPO_ROOT}/scripts/rust/hash-source.sh")"
      manifest_hash="$(jq -r '.source_hash_sha256' "${tmp_manifest}")"
      if [[ "${local_hash}" != "${manifest_hash}" ]]; then
        echo "Local source hash (${local_hash}) does not match manifest (${manifest_hash})." >&2
        echo "Do not trust this release. Investigate before using the artifact." >&2
        exit 1
      fi

      if [[ -f "${pin_file}" ]] && cmp -s "${tmp_manifest}" "${pin_file}"; then
        echo "rust/Pin.json already matches the published manifest. Nothing to commit."
        exit 0
      fi

      mv "${tmp_manifest}" "${pin_file}"
      trap - EXIT

      git add "${pin_file}"
      git commit -m "Pin Rust XCFramework to tokenizers-rust-${VERSION}"

      echo
      echo "Committed rust/Pin.json bump to tokenizers-rust-${VERSION}."
      echo "Push the branch and open or update the PR when ready."
      exit 0
    fi

    echo "Rust artifact publish failed." >&2
    echo "Run: ${run_url}" >&2
    exit 1
  fi

  sleep 10
done

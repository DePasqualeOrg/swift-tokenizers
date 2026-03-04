# Rust Release Process

This document covers the release flow for package versions that use the Rust backend.

Rust maintainer scripts live under:

- `scripts/rust/build/` for local artifact build and packaging helpers
- `scripts/rust/release/` for release, manifest-update, and tagging helpers

## Why this is two-stage

The package release and the Rust XCFramework release cannot happen at the same time:

1. The Rust XCFramework must be published first.
2. `TokenizersRustBinary/Package.swift` must then be updated to the final asset URL and checksum.
3. Only after that should the package itself be tagged and released.

Do not tag the package version before the binary package manifest points at the final semver Rust artifact.

## Rust artifact release

Trigger the Rust artifact workflow for the package version you intend to release:

```bash
bash scripts/rust/release/publish-rust-release.sh 0.3.0 --ref main
```

This runs `.github/workflows/rust-release.yml`, which:

- builds `TokenizersRust.xcframework`
- validates the local XCFramework with `swift test --traits Rust`
- generates release notes from `Cargo.lock`
- publishes `tokenizers-rust-<version>`

Examples:

- `0.3.0-rc.1` publishes a prerelease
- `0.3.0` publishes a final release

By default, the script waits for the workflow to finish and prints the release URL on success. Pass `--no-wait` if you only want to trigger the workflow.

## Update the binary package manifest

After the Rust artifact release is published, repoint the binary package manifest:

```bash
bash scripts/rust/release/update-rust-binary-package.sh 0.3.0
```

That updates:

- `TokenizersRustBinary/Package.swift`

## Validate the package

Run both test suites after the manifest update:

```bash
swift test
swift test --traits Rust
```

## Tag the package release

Create the package tag only after the manifest update has been committed:

```bash
bash scripts/rust/release/tag-package-release.sh 0.3.0
git push origin main
git push origin 0.3.0
```

If you want a GitHub release for the package tag itself, create it after the tag has been pushed.

## Cleanup

After the semver Rust artifact is live and the package manifest points to it, old transitional Rust artifact releases can be deleted.

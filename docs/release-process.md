# Release Process

`swift-tokenizers` has two linked releases:

- the Rust binary artifact release, tagged as `tokenizers-rust-<version>`
- the package release, tagged as `<version>`

The package release must come after the Rust artifact release, because `Package.swift`
needs to point at the published binary asset URL and checksum for that same version.

## Rust Artifact Release

1. Publish the Rust XCFramework artifact:

   ```bash
   bash scripts/rust/release/publish-rust-release.sh <version> --ref main
   ```

2. Update the root package manifest to the published artifact:

   ```bash
   bash scripts/rust/release/update-rust-binary-package.sh <version>
   ```

3. Validate both package modes:

   ```bash
   swift test
   swift test --traits Rust
   ```

4. Commit and push the `Package.swift` update to `main`.

## Package Release

Once `main` contains the matching `Package.swift` update, publish the package release:

```bash
bash scripts/rust/release/publish-package-release.sh <version>
```

That script:

- verifies `Package.swift` points at `tokenizers-rust-<version>`
- verifies the Rust artifact release exists
- creates and pushes the package tag `<version>`
- creates the matching GitHub release

## Example

For `0.3.1`, the sequence is:

```bash
bash scripts/rust/release/publish-rust-release.sh 0.3.1 --ref main
bash scripts/rust/release/update-rust-binary-package.sh 0.3.1
swift test
swift test --traits Rust
git commit -am "Point Rust Artifact At 0.3.1 Release"
git push origin main
bash scripts/rust/release/publish-package-release.sh 0.3.1
```

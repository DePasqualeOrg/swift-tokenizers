# Linux support

Reference for how the Swift package consumes the Rust backend on Linux, what the artifactbundle ships, and the constraints accepted on Apple platforms as a result.

## Supported platforms

The package's Rust backend is distributed as a single SE-0482 `staticLibrary` artifactbundle that carries Apple and Linux slices in one zip. Consumers add a single `.binaryTarget(url:checksum:)` dependency with no platform-conditional manifest logic.

| Platform | Triples |
|---|---|
| macOS 14+ | `arm64-apple-macosx`, `x86_64-apple-macosx` |
| iOS 17+ device | `arm64-apple-ios` |
| iOS 17+ simulator | `arm64-apple-ios-simulator`, `x86_64-apple-ios-simulator` |
| Linux (glibc) | `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu` |

The Linux slices are built on Ubuntu 22.04 so the resulting binaries are forward-compatible with the glibc shipped in newer Ubuntu LTS images. musl is not supported — Swift on Linux statically links against glibc symbols, and mixing a musl-built Rust archive with a glibc-Swift runtime is unsupported.

## Artifactbundle layout

```text
TokenizersRust.artifactbundle/
├── info.json
├── include/
│   ├── TokenizersRust.h
│   └── module.modulemap
├── apple-macos/
│   └── libtokenizers_rust.a       # fat: arm64 + x86_64
├── apple-ios-device/
│   └── libtokenizers_rust.a       # arm64 only
├── apple-ios-simulator/
│   └── libtokenizers_rust.a       # fat: arm64 + x86_64
├── linux-x86_64/
│   └── libtokenizers_rust.a
└── linux-aarch64/
    └── libtokenizers_rust.a
```

The `include/` directory ships once at the top level. The C ABI is platform-neutral, so every variant references the same header and modulemap via `staticLibraryMetadata.headerPaths` / `moduleMapPath`.

The bindgen-emitted modulemap from `uniffi-bindgen-swift` declares `use "Darwin"`, `use "_Builtin_stdbool"`, and `use "_Builtin_stdint"`. Those `use` directives reference clang modules that exist only on Apple and break the module load on Linux. `scripts/rust/build/strip-modulemap-uses.sh` drops them from the staged copy. The directives are advisory — clang resolves `<stdint.h>` etc. through the platform header search path on every target without them — so the strip is safe on Apple too.

## `info.json` shape

```json
{
  "schemaVersion": "1.0",
  "artifacts": {
    "TokenizersRust": {
      "type": "staticLibrary",
      "version": "<package version>",
      "variants": [
        {
          "path": "apple-macos/libtokenizers_rust.a",
          "supportedTriples": ["arm64-apple-macosx", "x86_64-apple-macosx"],
          "staticLibraryMetadata": {
            "headerPaths": ["include"],
            "moduleMapPath": "include/module.modulemap"
          }
        }
        // ... one variant per slice
      ]
    }
  }
}
```

`supportedTriples` are LLVM-style triples. Rust's target names do not always match what SwiftPM compares against at variant-selection time, so the strings in `info.json` are deliberately the LLVM forms:

| Rust target (`--target`) | LLVM triple (`supportedTriples`) |
|---|---|
| `aarch64-apple-darwin` | `arm64-apple-macosx` |
| `x86_64-apple-darwin` | `x86_64-apple-macosx` |
| `aarch64-apple-ios` | `arm64-apple-ios` |
| `aarch64-apple-ios-sim` | `arm64-apple-ios-simulator` |
| `x86_64-apple-ios` | `x86_64-apple-ios-simulator` |
| `x86_64-unknown-linux-gnu` | `x86_64-unknown-linux-gnu` |
| `aarch64-unknown-linux-gnu` | `aarch64-unknown-linux-gnu` |

`scripts/rust/build/assemble-artifactbundle.sh` performs this translation when emitting `info.json`.

## Linux linker settings

SE-0482 artifactbundles do not propagate transitive link dependencies — the Rust `staticlib` references symbols from libc satellites that resolve through libSystem on Apple but must be linked explicitly on Linux. `Package.swift` declares those on `TokenizersFFI` with `condition: .when(platforms: [.linux])`:

```swift
linkerSettings: ["dl", "pthread", "m", "rt", "util", "gcc_s"].map {
    .linkedLibrary($0, .when(platforms: [.linux]))
}
```

`libgcc_s` is normally auto-linked by the GCC linker driver, but some toolchain configurations skip it. The crate's `cargo rustc --crate-type staticlib -- --print=native-static-libs` output includes it, so it is declared explicitly.

When the Rust crate's dependency graph changes, recompute the canonical list:

```bash
cargo rustc \
  --manifest-path rust/Cargo.toml \
  --crate-type staticlib \
  --target aarch64-unknown-linux-gnu \
  -- --print=native-static-libs
```

`libc` is always implicit, so it does not need a `.linkedLibrary` entry.

## Privacy manifest

`tokenizers_rust` reads tokenizer JSON and sidecars from disk via `std::fs`, which compiles down to `fstat`/`lstat` calls. Apple lists those under the File Timestamp required-reason API category, so any iOS/macOS app shipping this package must declare a reason.

Artifactbundles cannot vend resources, so `Sources/TokenizersFFI/Resources/PrivacyInfo.xcprivacy` is staged as a SwiftPM resource on the `TokenizersFFI` target instead of inside the artifact itself.

Apple's reason codes for File Timestamp split between container-resident files and user-granted-access files, and `AutoTokenizer.from(directory:)` accepts an arbitrary directory URL — meaning a host app might be loading tokenizer files from its app/app-group/CloudKit container or from a user-picked location via a document picker. The manifest declares both reasons so the host app inherits coverage for whichever pattern it uses:

| Code | Use |
|---|---|
| `C617.1` | Files inside the app's container, app-group container, or CloudKit container. |
| `3B52.1` | Files or directories the user has specifically granted the app access to. |

## Constraints on Apple platforms

Switching from XCFramework to artifactbundle has three knock-on effects:

- **Swift 6.2 / Xcode 26 floor.** SE-0482's `staticLibrary` artifact type landed in Swift 6.2; older Xcode versions cannot resolve the package.
- **Privacy manifest is a SwiftPM resource, not artifact-resident.** `Sources/TokenizersFFI/Resources/PrivacyInfo.xcprivacy` (declared as a `.process(...)` resource on `TokenizersFFI`) replaces the per-slice manifest an XCFramework could carry. Functionally equivalent for App Store consumers, but it does mean the artifactbundle alone is not a complete iOS/macOS-shippable unit; the wrapping Swift target is.
- **No legacy-Xcode drag-and-drop consumption.** Artifactbundles are a SwiftPM construct; an Xcode project that doesn't use SwiftPM cannot consume the package.

## Known issues

### URLSession redirect crash on Swift 6.2.x / 6.3.x + Linux

`URLSession` requests that follow an HTTP redirect crash deterministically in `_HTTPURLProtocol.configureEasyHandle` with `FoundationNetworking/EasyHandle.swift: Fatal error: 'try!' expression unexpectedly raised an error: Error Domain=libcurl.Easy Code=43 "(null)"`. `Code=43` is `CURLE_BAD_FUNCTION_ARGUMENT`. Hugging Face Hub triggers it on every download because the API hosts redirect to a CDN.

This repository's Swift code does not use `URLSession`; the crash only affects callers that do, including `HubClient` from the `swift-hf-api` test dependency that some of the package's tests use.

**Upstream tracking:**

- Issue: <https://github.com/swiftlang/swift-corelibs-foundation/issues/5445>
- Fix: <https://github.com/swiftlang/swift-corelibs-foundation/pull/5448> (merged 2026-04-08)

The fix removes a `preferredReceiveBufferSize` call that libcurl 8.5.0 (Ubuntu 24.04 `noble`, Debian `bookworm`) rejects on a handle with an existing buffer. Reproduces on Ubuntu 22.04 (`jammy`, libcurl 7.81) too, so the fix is needed regardless of the runner's libcurl version. The fix landed on `main` and `release/6.4.x` only — not backported to `release/6.2.x` or `release/6.3.x` — so the first user-visible Swift release containing it is **Swift 6.4**.

The package's PR CI's Linux job filters its test invocation to the offline subset (`AlgorithmTests` + `UniffiSpikeTests`) for this reason. The filter can be removed once *either* condition holds: the workflow's `swift:6.2.3` container is upgraded to `swift:6.4` or later (so the FoundationNetworking fix is on the runtime stack), or `swift-hf-api` migrates its `Hub/` subsystem to wrap the Rust [`hf-hub`](https://github.com/huggingface/hf-hub) crate (so `HubClient` no longer goes through FoundationNetworking on Linux at all). Either change makes the full suite pass on Linux.

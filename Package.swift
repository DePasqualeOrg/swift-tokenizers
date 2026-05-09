// swift-tools-version: 6.2

import PackageDescription

// Pinned artifactbundle for the Rust backend. These mirror `rust/Pin.json` and are
// kept in sync by scripts/rust/release/cut-release.sh. Inlined here rather than
// read from Pin.json because manifest-eval file I/O is unreliable for URL-based
// dependency consumers (both `Context.packageDirectory` and `#filePath` return
// synthetic paths during dep evaluation).
let tokenizersRustArtifactBundleURL =
    "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/tokenizers-rust-0.5.0/TokenizersRust-0.5.0.xcframework.zip"
let tokenizersRustArtifactBundleChecksum =
    "a6f475c34de051e090bd14c072b393d33309a2de5f89ac49fc3bf0c2287151bd"

let docsEnabled = Context.environment["TOKENIZERS_ENABLE_DOCS"] == "1"
let localRustArtifactPath = Context.environment["TOKENIZERS_RUST_LOCAL_ARTIFACTBUNDLE_PATH"]

var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.3.2")
]

// The Benchmarks target pulls in mlx-swift-lm, which is macOS-only and
// transitively requires Metal and Accelerate. Gate the dep + target on macOS
// + opt-in env var so plain `swift test` on macOS doesn't compile an
// unbuildable graph and iOS consumers don't trip over the macOS-only dep.
// Xcode users who want to see Benchmarks in the test navigator must launch
// Xcode with the env var set (e.g. `launchctl setenv TOKENIZERS_ENABLE_BENCHMARKS 1`).
let benchmarksEnabled = Context.environment["TOKENIZERS_ENABLE_BENCHMARKS"] == "1"
#if os(macOS)
if benchmarksEnabled {
    packageDependencies.append(
        .package(url: "https://github.com/DePasqualeOrg/mlx-swift-lm.git", branch: "main")
    )
}
#endif

if docsEnabled {
    packageDependencies.append(
        .package(url: "https://github.com/swiftlang/swift-docc-plugin", from: "1.4.0")
    )
}

let tokenizersRustTarget: Target =
    if let localRustArtifactPath {
        // Used by the Rust release workflow to validate a freshly built artifactbundle before publishing.
        // The override must point at an unzipped directory whose name ends in `.artifactbundle`; SwiftPM
        // uses the suffix to discriminate, so a `.zip` path will not work.
        .binaryTarget(name: "TokenizersRust", path: localRustArtifactPath)
    } else {
        .binaryTarget(
            name: "TokenizersRust",
            url: tokenizersRustArtifactBundleURL,
            checksum: tokenizersRustArtifactBundleChecksum
        )
    }

var packageTargets: [Target] = [
    tokenizersRustTarget,
    // Holds the UniFFI-generated Swift wrapper. Not in `products` — its public
    // declarations stay invisible to consumers of the `Tokenizers` library so
    // the public Swift API surface is unaffected. The committed wrapper lives
    // at `Sources/TokenizersFFI/Generated/<wrapper>.swift` and is regenerated
    // by `scripts/rust/regenerate-wrapper.sh`.
    //
    // `linkerSettings` covers symbols the Rust staticlib pulls in from libc
    // satellites. SE-0482 artifactbundles do not propagate transitive link
    // dependencies, and on Apple these resolve through libSystem without
    // explicit settings, so the list is gated to Linux only. `gcc_s` is
    // normally auto-linked by gcc but appears in
    // `cargo rustc -- --print=native-static-libs` output, so we declare it too.
    //
    // The `Resources` directory carries `PrivacyInfo.xcprivacy`. The Rust
    // backend reads tokenizer JSON via `std::fs`, which compiles down to
    // `fstat`/`lstat` calls — those fall under Apple's File Timestamp
    // required-reason API category, so the manifest declares both the
    // app-container reason (`C617.1`) and the user-granted-access reason
    // (`3B52.1`) so the host app inherits coverage for both common loading
    // patterns of `AutoTokenizer.from(directory:)`.
    .target(
        name: "TokenizersFFI",
        dependencies: [
            .target(name: "TokenizersRust")
        ],
        path: "Sources/TokenizersFFI",
        resources: [.process("Resources")],
        linkerSettings: ["dl", "pthread", "m", "rt", "util", "gcc_s"].map {
            .linkedLibrary($0, .when(platforms: [.linux]))
        }
    ),
    .target(
        name: "Tokenizers",
        dependencies: [
            .target(name: "TokenizersFFI")
        ],
        path: "Sources/Tokenizers"
    ),
    .testTarget(
        name: "TokenizersTests",
        dependencies: [
            "Tokenizers",
            "TokenizersFFI",
            .product(name: "HFAPI", package: "swift-hf-api"),
        ],
        resources: [.process("Resources")]
    ),
]

#if os(macOS)
if benchmarksEnabled {
    packageTargets.append(
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "Tokenizers",
                .product(name: "HFAPI", package: "swift-hf-api"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        )
    )
}
#endif

let package = Package(
    name: "swift-tokenizers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Tokenizers", targets: ["Tokenizers"])
    ],
    dependencies: packageDependencies,
    targets: packageTargets
)

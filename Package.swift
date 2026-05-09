// swift-tools-version: 6.1

import PackageDescription

// Pinned XCFramework for the Rust backend. These mirror `rust/Pin.json` and are
// kept in sync by scripts/rust/release/cut-release.sh. Inlined here rather than
// read from Pin.json because manifest-eval file I/O is unreliable for URL-based
// dependency consumers (both `Context.packageDirectory` and `#filePath` return
// synthetic paths during dep evaluation).
let tokenizersRustXCFrameworkURL =
    "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/tokenizers-rust-0.5.0/TokenizersRust-0.5.0.xcframework.zip"
let tokenizersRustXCFrameworkChecksum =
    "a6f475c34de051e090bd14c072b393d33309a2de5f89ac49fc3bf0c2287151bd"

let docsEnabled = Context.environment["TOKENIZERS_ENABLE_DOCS"] == "1"
let localRustArtifactPath = Context.environment["TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH"]

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
        // Used by the Rust release workflow to validate a freshly built XCFramework before publishing.
        .binaryTarget(name: "TokenizersRust", path: localRustArtifactPath)
    } else {
        .binaryTarget(
            name: "TokenizersRust",
            url: tokenizersRustXCFrameworkURL,
            checksum: tokenizersRustXCFrameworkChecksum
        )
    }

var packageTargets: [Target] = [
    tokenizersRustTarget,
    // Holds the UniFFI-generated Swift wrapper. Not in `products` — its public
    // declarations stay invisible to consumers of the `Tokenizers` library so
    // the public Swift API surface is unaffected. The committed wrapper lives
    // at `Sources/TokenizersFFI/Generated/<wrapper>.swift` and is regenerated
    // by `scripts/rust/regenerate-wrapper.sh`.
    .target(
        name: "TokenizersFFI",
        dependencies: [
            .target(name: "TokenizersRust")
        ],
        path: "Sources/TokenizersFFI"
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

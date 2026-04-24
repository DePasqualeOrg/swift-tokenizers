// swift-tools-version: 6.1

import PackageDescription

// Pinned XCFramework for the Rust backend. These mirror `rust/Pin.json` and are
// kept in sync by scripts/rust/release/cut-release.sh. Inlined here rather than
// read from Pin.json because manifest-eval file I/O is unreliable for URL-based
// dependency consumers (both `Context.packageDirectory` and `#filePath` return
// synthetic paths during dep evaluation).
let tokenizersRustXCFrameworkURL =
    "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/tokenizers-rust-0.4.1/TokenizersRust-0.4.1.xcframework.zip"
let tokenizersRustXCFrameworkChecksum =
    "9b403e6053eefdcbb3a5aac62577467a4c1ae970e84df0ac28bd22c624bc0832"

let tokenizerCoreSources = [
    "BinaryDistinct.swift",
    "Config.swift",
    "Tokenizer.swift",
    "TokenizerRuntimeConfiguration.swift",
]

let tokenizerSwiftBackendSources = [
    "BPETokenizer.swift",
    "ByteEncoder.swift",
    "Decoder.swift",
    "Normalizer.swift",
    "PostProcessor.swift",
    "PreTokenizer.swift",
    "String+PreTokenization.swift",
    "TokenLattice.swift",
    "Trie.swift",
    "UnigramTokenizer.swift",
    "WordLevelTokenizer.swift",
    "WordPieceTokenizer.swift",
    "YYJSONParser.swift",
    "SwiftTokenizerBackend.swift",
]

let tokenizerRustBackendSources = [
    "RustBackedTokenizer.swift"
]

let tokenizerDirectorySources =
    tokenizerCoreSources
    + tokenizerSwiftBackendSources
    + tokenizerRustBackendSources

let docsEnabled = Context.environment["TOKENIZERS_ENABLE_DOCS"] == "1"
let localRustArtifactPath = Context.environment["TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH"]

// xcodebuild has no `--traits` flag; `TOKENIZERS_BACKEND=Rust` flips the default trait
// at manifest-eval time. `swift test --traits` and Xcode's Package Traits UI are unaffected.
let defaultBackendTrait = Context.environment["TOKENIZERS_BACKEND"] == "Rust" ? "Rust" : "Swift"

func excludedTokenizerSources(keeping sources: [String]) -> [String] {
    tokenizerDirectorySources.filter { !sources.contains($0) }
}

var packageDependencies: [Package.Dependency] = [
    .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
    .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.2.0"),
]

// The Benchmarks target pulls in mlx-swift-lm, which transitively requires Metal
// and Accelerate. Gate the dep + target on macOS + opt-in env var so Linux (and
// plain `swift test` on macOS) don't compile an unbuildable graph. Xcode users
// who want to see Benchmarks in the test navigator must launch Xcode with the
// env var set (e.g. `launchctl setenv TOKENIZERS_ENABLE_BENCHMARKS 1`).
let benchmarksEnabled = Context.environment["TOKENIZERS_ENABLE_BENCHMARKS"] == "1"
#if os(macOS)
if benchmarksEnabled {
    packageDependencies.append(
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", from: "3.31.3")
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
    .target(
        name: "TokenizersCore",
        dependencies: [],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerCoreSources),
        sources: tokenizerCoreSources
    ),
    .target(
        name: "TokenizersSwiftBackend",
        dependencies: [
            "TokenizersCore",
            .product(name: "Jinja", package: "swift-jinja", condition: .when(traits: ["Swift"])),
            .product(name: "yyjson", package: "yyjson", condition: .when(traits: ["Swift"])),
        ],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerSwiftBackendSources),
        sources: tokenizerSwiftBackendSources,
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"]))
        ]
    ),
    .target(
        name: "TokenizersRustBackend",
        dependencies: [
            "TokenizersCore",
            .target(name: "TokenizersRust", condition: .when(traits: ["Rust"])),
        ],
        path: "Sources/Tokenizers",
        exclude: excludedTokenizerSources(keeping: tokenizerRustBackendSources),
        sources: tokenizerRustBackendSources,
        swiftSettings: [
            .define("Rust", .when(traits: ["Rust"]))
        ]
    ),
    .target(
        name: "Tokenizers",
        dependencies: [
            "TokenizersCore",
            .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
            .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
        ],
        path: "Sources/TokenizersFacade",
        swiftSettings: {
            var settings: [SwiftSetting] = [
                .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
                .define("Rust", .when(traits: ["Rust"])),
            ]
            if docsEnabled {
                // swift-docc-plugin doesn't propagate traits to symbol-graph sub-builds, so
                // both backends compile at once. BackendSelection.swift drops the mutex guard
                // when this define is set.
                settings.append(.define("TOKENIZERS_DOCS_BUILD"))
            }
            return settings
        }()
    ),
    .testTarget(
        name: "TokenizersTests",
        dependencies: [
            "Tokenizers",
            "TokenizersCore",
            .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
            .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
            .product(name: "HFAPI", package: "swift-hf-api"),
        ],
        resources: [.process("Resources")],
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
            .define("Rust", .when(traits: ["Rust"])),
        ]
    ),
]

#if os(macOS)
if benchmarksEnabled {
    packageTargets.append(
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "Tokenizers",
                "TokenizersCore",
                .target(name: "TokenizersSwiftBackend", condition: .when(traits: ["Swift"])),
                .target(name: "TokenizersRustBackend", condition: .when(traits: ["Rust"])),
                .product(name: "HFAPI", package: "swift-hf-api"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [
                .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
                .define("Rust", .when(traits: ["Rust"])),
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
    traits: [
        .default(enabledTraits: [defaultBackendTrait]),
        .trait(name: "Swift"),
        .trait(name: "Rust"),
    ],
    dependencies: packageDependencies,
    targets: packageTargets
)

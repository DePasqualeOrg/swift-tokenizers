// swift-tools-version: 6.1

import PackageDescription

let tokenizerCoreSources = [
    "BinaryDistinct.swift",
    "Config.swift",
    "Tokenizer.swift",
    "TokenizerCompatibility.swift",
    "TokenizerRuntimeConfiguration.swift",
]

let tokenizerSwiftBackendSources = [
    "BPETokenizer.swift",
    "BertTokenizer.swift",
    "ByteEncoder.swift",
    "Decoder.swift",
    "Normalizer.swift",
    "PostProcessor.swift",
    "PreTokenizer.swift",
    "String+PreTokenization.swift",
    "TokenLattice.swift",
    "Trie.swift",
    "UnigramTokenizer.swift",
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

let benchmarksEnabled = Context.environment["TOKENIZERS_ENABLE_BENCHMARKS"] == "1"

func excludedTokenizerSources(keeping sources: [String]) -> [String] {
    tokenizerDirectorySources.filter { !sources.contains($0) }
}

var packageDependencies: [Package.Dependency] = [
    .package(path: "TokenizersRustBinary"),
    .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
    .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
    .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.2.0"),
]

if benchmarksEnabled {
    packageDependencies.append(
        .package(
            // TODO: Switch to a major version pin once mlx-swift-lm publishes a new major release that includes these APIs.
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "8c9dd6391139242261bcf27d253c326f9cf2d567"
        )
    )
}

var packageTargets: [Target] = [
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
            .product(name: "TokenizersRust", package: "TokenizersRustBinary", condition: .when(traits: ["Rust"])),
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
        swiftSettings: [
            .define("TOKENIZERS_SWIFT_BACKEND", .when(traits: ["Swift"])),
            .define("Rust", .when(traits: ["Rust"])),
        ]
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

let package = Package(
    name: "swift-tokenizers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Tokenizers", targets: ["Tokenizers"])
    ],
    traits: [
        .default(enabledTraits: ["Swift"]),
        .trait(name: "Swift"),
        .trait(name: "Rust"),
    ],
    dependencies: packageDependencies,
    targets: packageTargets
)

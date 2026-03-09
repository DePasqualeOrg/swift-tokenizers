// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "swift-tokenizers",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "Tokenizers", targets: ["Tokenizers"])
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.0.0"),
        .package(url: "https://github.com/ibireme/yyjson.git", exact: "0.12.0"),
        .package(url: "https://github.com/DePasqualeOrg/swift-hf-api.git", from: "0.2.0"),
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm.git",
            revision: "8c9dd6391139242261bcf27d253c326f9cf2d567"
        ),
    ],
    targets: [
        .target(
            name: "Tokenizers",
            dependencies: [
                .product(name: "Jinja", package: "swift-jinja"),
                .product(name: "yyjson", package: "yyjson"),
            ]
        ),
        .testTarget(
            name: "Benchmarks",
            dependencies: [
                "Tokenizers",
                .product(name: "HFAPI", package: "swift-hf-api"),
                .product(name: "BenchmarkHelpers", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .testTarget(
            name: "TokenizersTests",
            dependencies: [
                "Tokenizers",
                .product(name: "HFAPI", package: "swift-hf-api"),
            ],
            resources: [.process("Resources")]
        ),
    ]
)

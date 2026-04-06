// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "TokenizersRustBinary",
    products: [
        .library(name: "TokenizersRust", targets: ["TokenizersRust"])
    ],
    targets: [
        .binaryTarget(
            name: "TokenizersRust",
            url: "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/tokenizers-rust-rename-rust-artifact-1/TokenizersRust-main-3.xcframework.zip",
            checksum: "ec56924f0ed8493937da06496f09a680ed81803fd20ef33769538a95ec85ff77"
        )
    ]
)

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
            path: "../Binaries/TokenizersRust.xcframework"
        )
    ]
)

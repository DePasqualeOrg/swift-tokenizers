// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "RustCorePackage",
    products: [
        .library(name: "TokenizersRustBinary", targets: ["TokenizersRustCore"])
    ],
    targets: [
        .binaryTarget(
            name: "TokenizersRustCore",
            path: "../Binaries/TokenizersRustCore.xcframework"
        )
    ]
)

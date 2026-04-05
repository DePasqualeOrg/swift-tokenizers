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
            url: "https://github.com/DePasqualeOrg/swift-tokenizers/releases/download/rust-core-add-rust-backend-1/TokenizersRustCore-add-rust-backend-1.xcframework.zip",
            checksum: "03052342bcc47e3228655c696f36b9c1884807618339a724c98c05d57e7e6ad4"
        )
    ]
)

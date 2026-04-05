Swift Tokenizers is a streamlined and optimized fork of Swift Transformers that focuses solely on tokenizer functionality. It has no dependency on the Hugging Face Hub: tokenizers are simply loaded from a directory, and downloading is handled separately.

The package supports two backends behind the same Swift API:

- `Swift` is the default backend. It is pure Swift and requires no Rust artifact.
- `Rust` is an opt-in backend that uses a prebuilt Rust core for faster tokenization and decoding on supported Apple platforms.

Refer to the [Benchmarks](#benchmarks) section to compare the performance of Swift Tokenizers and Swift Transformers.

## Package setup

Swift Tokenizers uses Swift package traits and requires Swift 6.1 or newer.

### Default pure Swift backend

If you add the package without specifying traits, the default `Swift` trait is enabled:

```swift
dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.1.0")
]
```

### Opt in to the Rust backend

To build with the Rust backend instead of the default Swift backend, enable only the `Rust` trait:

```swift
dependencies: [
    .package(
        url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
        from: "0.1.0",
        traits: ["Rust"]
    )
]
```

The package traits are intentionally mutually exclusive:

- default dependency declaration: enables the `Swift` backend
- `traits: ["Rust"]`: enables the `Rust` backend only

Do not combine `.defaults` and `"Rust"` for this package.

### Building the local Rust artifact

The Rust backend currently expects the XCFramework to be present locally when building from a checkout of this repository:

```bash
bash scripts/build-rust-core-xcframework.sh
```

The repository also includes:

- `scripts/package-rust-core-release.sh` to zip the XCFramework and compute the SwiftPM checksum
- `.github/workflows/rust-core-release.yml` to build the release artifact in CI and publish it as a GitHub prerelease asset

## Examples

### Loading a tokenizer

Load a tokenizer from a local directory containing `tokenizer.json` and any relevant sidecar files such as `tokenizer_config.json`, `config.json`, and `chat_template.jinja`:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(directory: localDirectory)
```

### Encoding and decoding

```swift
let tokens = tokenizer.encode(text: "The quick brown fox")
let text = tokenizer.decode(tokens: tokens)
```

### Chat templates

```swift
let messages: [[String: any Sendable]] = [
    ["role": "user", "content": "Describe the Swift programming language."],
]
let encoded = try tokenizer.applyChatTemplate(messages: messages)
let decoded = tokenizer.decode(tokens: encoded)
```

### Tool calling

```swift
let weatherTool = [
    "type": "function",
    "function": [
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": [
            "type": "object",
            "properties": ["location": ["type": "string", "description": "City and state"]],
            "required": ["location"]
        ]
    ]
]

let tokens = try tokenizer.applyChatTemplate(
    messages: [["role": "user", "content": "What's the weather in Paris?"]],
    tools: [weatherTool]
)
```

## Migration from Swift Transformers

This library focuses solely on tokenization. For downloading models from the Hugging Face Hub, use [Swift Hugging Face](https://github.com/huggingface/swift-huggingface).

### Package dependency

Replace `swift-transformers` with `swift-tokenizers` in your `Package.swift`. The `Transformers` product no longer exists – use the `Tokenizers` product directly:

```swift
// Before
.package(url: "https://github.com/huggingface/swift-transformers.git", from: "..."),
// ...
.product(name: "Transformers", package: "swift-transformers"),

// After
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "..."),
// ...
.product(name: "Tokenizers", package: "swift-tokenizers"),
```

If you want the Rust backend, enable the `Rust` trait on the package dependency:

```swift
.package(
    url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
    from: "...",
    traits: ["Rust"]
),
```

### Loading tokenizers

Download model files separately, then load from a local directory.

```swift
// Before
let tokenizer = try await AutoTokenizer.from(pretrained: "model-name", hubApi: hub)
let tokenizer = try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)

// After (download tokenizer files to directory first)
let tokenizer = try await AutoTokenizer.from(directory: directory)
```

## Benchmarks

The benchmarks use tests from MLX Swift LM and can be run from this package in Xcode.

Set `TOKENIZERS_ENABLE_BENCHMARKS=1` to include the benchmark target in the package graph, then set `RUN_BENCHMARKS=1` in the test scheme environment to run the benchmark suite.

From the command line, use release builds for accurate numbers:

```bash
TOKENIZERS_ENABLE_BENCHMARKS=1 RUN_BENCHMARKS=1 swift test -c release --filter Benchmarks
TOKENIZERS_ENABLE_BENCHMARKS=1 RUN_BENCHMARKS=1 swift test -c release --traits Rust --filter Benchmarks
```

These results were observed on an M3 MacBook Pro.

| Benchmark | Swift Tokenizers median | Swift Transformers median | Swift Tokenizers Performance |
| --- | ---: | ---: | --- |
| Tokenizer load | 289.6 ms | 1004.6 ms | 3.47x faster |
| Tokenization | 53.0 ms | 105.8 ms | 2.00x faster |
| Decoding | 28.9 ms | 48.4 ms | 1.67x faster |
| LLM load | 318.8 ms | 1033.5 ms | 3.24x faster |
| VLM load | 367.9 ms | 1081.5 ms | 2.94x faster |
| Embedding load | 310.7 ms | 1023.5 ms | 3.29x faster |

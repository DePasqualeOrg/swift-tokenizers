Swift Tokenizers is a high-performance tokenizer library aligned with Python Transformers v5 and Rust Tokenizers. Unlike Swift Transformers, it focuses solely on tokenizer functionality and has no dependency on the Hugging Face Hub. It offers significantly faster model loading and tokenization performance in Swift and further performance improvements when using the optional Rust backend.

Refer to the [Benchmarks](#benchmarks) section to compare the performance of Swift Tokenizers and Swift Transformers.

Two backends are available using Swift package traits:

| | Swift (default) | Rust (opt-in) |
|---|---|---|
| Tokenization | Swift | [tokenizers](https://github.com/huggingface/tokenizers) |
| Chat templates | [Swift Jinja](https://github.com/huggingface/swift-jinja) | [MiniJinja](https://github.com/mitsuhiko/minijinja) |
| JSON parsing | [yyjson](https://github.com/ibireme/yyjson) (C) | [serde](https://github.com/serde-rs/serde) |

The opt-in `Rust` trait links a Rust binary and excludes the corresponding Swift implementations for even faster performance than the optimized Swift backend.

## Package setup

Swift Tokenizers uses Swift package traits and requires Swift 6.1 or newer.

### Default Swift backend

```swift
dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.3.2", traits: ["Swift"])
]
```

### Opt in to the Rust backend

To build with the Rust backend instead of the default Swift backend, enable only the `Rust` trait:

```swift
dependencies: [
    .package(
        url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
        from: "0.3.2",
        traits: ["Rust"]
    )
]
```

The package traits are intentionally mutually exclusive:

- default dependency declaration: enables the `Swift` backend
- `traits: ["Rust"]`: enables the `Rust` backend only

Do not combine `.defaults` and `"Rust"` for this package.

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

This library focuses solely on tokenization. The separate [Swift HF API](https://github.com/DePasqualeOrg/swift-hf-api) is an optimized client for the Hugging Face Hub API.

### Package dependency

Replace `swift-transformers` with `swift-tokenizers` in your `Package.swift`. The `Transformers` product no longer exists – use the `Tokenizers` product directly:

```swift
// Before
.package(url: "https://github.com/huggingface/swift-transformers.git", from: "..."),
// ...
.product(name: "Transformers", package: "swift-transformers"),

// After
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "...", traits: ["Swift"]),
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

| | Swift Transformers | Swift backend | | Rust backend | |
| --- | ---: | ---: | --- | ---: | --- |
| Tokenizer load | 399.3 ms | 176.1 ms | 2.3x faster | 164.5 ms | 2.4x faster |
| Tokenization | 48.4 ms | 23.0 ms | 2.1x faster | 3.5 ms | 13.8x faster |
| Decoding | 30.9 ms | 13.3 ms | 2.3x faster | 3.7 ms | 8.4x faster |
| LLM load | 409.7 ms | 189.5 ms | 2.2x faster | 184.5 ms | 2.2x faster |
| VLM load | 441.6 ms | 235.2 ms | 1.9x faster | 223.1 ms | 2.0x faster |
| Embedding load | 412.0 ms | 191.5 ms | 2.2x faster | 191.6 ms | 2.2x faster |

These results were observed on an M3 MacBook Pro using Swift Tokenizers `7e5ea0d`, Swift Transformers [`1.3.0`](https://github.com/huggingface/swift-transformers/releases/tag/1.3.0), and MLX Swift LM `8c9dd63`.

### Running benchmarks

The benchmarks use tests from MLX Swift LM and can be run from this package in Xcode. Set `TOKENIZERS_ENABLE_BENCHMARKS=1` to include the benchmark target in the package graph and enable the benchmark suite.

From the command line, use release builds for accurate numbers. The model loading benchmarks (LLM, VLM, embedding) require Metal, which is only available through `xcodebuild`. However, `xcodebuild` does not support package traits, so `swift test` is needed to run benchmarks with the Rust backend.

```bash
# Full suite (requires Metal)
TOKENIZERS_ENABLE_BENCHMARKS=1 TEST_RUNNER_TOKENIZERS_ENABLE_BENCHMARKS=1 xcodebuild test -scheme Benchmarks -configuration Release -destination 'platform=macOS,arch=arm64'

# Tokenizer benchmarks only
TOKENIZERS_ENABLE_BENCHMARKS=1 swift test -c release --filter Benchmarks

# Tokenizer benchmarks with Rust backend
TOKENIZERS_ENABLE_BENCHMARKS=1 swift test -c release --traits Rust --filter Benchmarks
```

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
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.4.2", traits: ["Swift"])
]
```

### Opt in to the Rust backend

To build with the Rust backend instead of the default Swift backend, enable only the `Rust` trait:

```swift
dependencies: [
    .package(
        url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
        from: "0.4.2",
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
| Tokenizer load | 399.3 ms | 178.5 ms | 2.2x faster | 170.5 ms | 2.3x faster |
| Tokenization | 48.4 ms | 24.4 ms | 2.0x faster | 3.7 ms | 13.1x faster |
| Decoding | 30.9 ms | 14.6 ms | 2.1x faster | 3.9 ms | 7.9x faster |
| LLM load | 409.7 ms | 195.5 ms | 2.1x faster | 192.5 ms | 2.1x faster |
| VLM load | 441.6 ms | 233.8 ms | 1.9x faster | 233.8 ms | 1.9x faster |
| Embedding load | 412.0 ms | 204.2 ms | 2.0x faster | 198.8 ms | 2.1x faster |

These results were observed on an M3 MacBook Pro using Swift Tokenizers [`0.4.2`](https://github.com/DePasqualeOrg/swift-tokenizers/releases/tag/0.4.2), Swift Transformers [`1.3.0`](https://github.com/huggingface/swift-transformers/releases/tag/1.3.0), and MLX Swift LM [`3.31.3`](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.31.3).

### Running benchmarks

The benchmarks use tests from MLX Swift LM and are gated behind `TOKENIZERS_ENABLE_BENCHMARKS=1` so that ordinary consumers don't pull `mlx-swift-lm` (which requires Metal/Accelerate and doesn't build on Linux) into their dependency graph. Set the env var before evaluating the package to include the benchmark target.

**In Xcode**: the env var must be present when Xcode resolves the package, which happens on launch. The easiest persistent option is `launchctl setenv TOKENIZERS_ENABLE_BENCHMARKS 1` (run once, then reopen Xcode). To run with the Rust backend, enable the `Rust` trait under **File → Packages → Package Traits…**.

**From the command line**: use release builds for accurate numbers. Model loading benchmarks (LLM, VLM, embedding) require Metal, which is only available through `xcodebuild`. `xcodebuild` has no `--traits` flag, so `TOKENIZERS_BACKEND=Rust` flips the default trait at manifest-evaluation time to drive the Rust backend.

```bash
# Full suite, Swift backend (requires Metal)
TOKENIZERS_ENABLE_BENCHMARKS=1 xcodebuild test -scheme Benchmarks -destination 'platform=macOS,arch=arm64'

# Full suite, Rust backend (requires Metal)
TOKENIZERS_ENABLE_BENCHMARKS=1 TOKENIZERS_BACKEND=Rust xcodebuild test -scheme Benchmarks -destination 'platform=macOS,arch=arm64'

# Tokenizer benchmarks only, Swift backend
TOKENIZERS_ENABLE_BENCHMARKS=1 swift test -c release --filter Benchmarks

# Tokenizer benchmarks only, Rust backend
TOKENIZERS_ENABLE_BENCHMARKS=1 swift test -c release --traits Rust --filter Benchmarks
```

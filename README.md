Swift Tokenizers is a high-performance Swift wrapper around Hugging Face's Rust `tokenizers` crate. Unlike Swift Transformers, it focuses solely on tokenization and has no dependency on the Hugging Face Hub.

Refer to the [Benchmarks](#benchmarks) section to compare the performance of Swift Tokenizers and Swift Transformers.

## Package Setup

Swift Tokenizers requires Swift 6.1 or newer and supports Apple platforms only (macOS 14+, iOS 17+). The Rust backend is distributed as a prebuilt XCFramework, so Linux is not supported.

```swift
dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.5.0")
]
```

## Examples

### Loading a Tokenizer

Load a tokenizer from a local directory containing `tokenizer.json` and any relevant sidecar files such as `tokenizer_config.json`, `config.json`, and `chat_template.jinja`:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(directory: localDirectory)
```

### Encoding and Decoding

Use `encode` when you only need token IDs:

```swift
let tokenIds = tokenizer.encode(text: "The quick brown fox")
let text = tokenizer.decode(tokenIds: tokenIds)
```

Use `encodeWithMetadata` when you need the richer encoding data, including token strings, masks, sequence indices, word indices, and offset spans:

```swift
let encoding = try tokenizer.encodeWithMetadata(text: "The quick brown fox")
let tokenIds = encoding.tokenIds
let firstTokenSpan = encoding.offsetSpan(forTokenIndex: 0)
```

### Chat Templates

```swift
let messages: [[String: any Sendable]] = [
    ["role": "user", "content": "Describe the Swift programming language."],
]
let encoded = try tokenizer.applyChatTemplate(messages: messages)
let decoded = tokenizer.decode(tokenIds: encoded)
```

### Tool Calling

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

## Migration From Swift Transformers

This library focuses solely on tokenization. The separate [Swift HF API](https://github.com/DePasqualeOrg/swift-hf-api) is an optimized client for the Hugging Face Hub API.

### Package Dependency

Replace `swift-transformers` with `swift-tokenizers` in your `Package.swift`. The `Transformers` product no longer exists, so use the `Tokenizers` product directly:

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

### Loading Tokenizers

Download model files separately, then load from a local directory.

```swift
// Before
let tokenizer = try await AutoTokenizer.from(pretrained: "model-name", hubApi: hub)
let tokenizer = try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)

// After (download tokenizer files to directory first)
let tokenizer = try await AutoTokenizer.from(directory: directory)
```

## Benchmarks

| | Swift Transformers | Swift Tokenizers | |
| --- | ---: | ---: | --- |
| Tokenizer load | 399.3 ms | 168.1 ms | 2.4x faster |
| Tokenization | 48.4 ms | 4.5 ms | 10.8x faster |
| Decoding | 30.9 ms | 3.9 ms | 7.9x faster |
| LLM load | 409.7 ms | 183.8 ms | 2.2x faster |
| VLM load | 441.6 ms | 225.8 ms | 2.0x faster |
| Embedding load | 412.0 ms | 197.7 ms | 2.1x faster |

These results were observed on an M3 MacBook Pro using Swift Tokenizers [`0.5.0`](https://github.com/DePasqualeOrg/swift-tokenizers/releases/tag/0.5.0), Swift Transformers [`1.3.0`](https://github.com/huggingface/swift-transformers/releases/tag/1.3.0), and MLX Swift LM [`3.31.3`](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.31.3).

### Running Benchmarks

The benchmarks use tests from MLX Swift LM and are gated behind `TOKENIZERS_ENABLE_BENCHMARKS=1` so that ordinary consumers do not pull `mlx-swift-lm` (which requires Metal/Accelerate and is macOS-only) into their dependency graph. Set the env var before evaluating the package to include the benchmark target.

**In Xcode**: the env var must be present when Xcode resolves the package, which happens on launch. The easiest persistent option is `launchctl setenv TOKENIZERS_ENABLE_BENCHMARKS 1` (run once, then reopen Xcode).

**From the command line**: use release builds for accurate numbers. Model loading benchmarks (LLM, VLM, embedding) require Metal, which is only available through `xcodebuild`.

```bash
# Full suite, requires Metal
TOKENIZERS_ENABLE_BENCHMARKS=1 xcodebuild test -scheme Benchmarks -destination 'platform=macOS,arch=arm64'

# Tokenizer benchmarks only
TOKENIZERS_ENABLE_BENCHMARKS=1 swift test -c release --filter Benchmarks
```

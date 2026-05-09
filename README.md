Swift Tokenizers is a high-performance Swift wrapper around Hugging Face's Rust `tokenizers` crate. Unlike Swift Transformers, it focuses solely on tokenization and has no dependency on the Hugging Face Hub.

Refer to the [Benchmarks](#benchmarks) section to compare the performance of Swift Tokenizers and Swift Transformers.

## Package Setup

Swift Tokenizers requires Swift 6.2 or newer (Xcode 26 on Apple platforms) and supports macOS 14+, iOS 17+, and Linux (`x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`). The Rust backend is distributed as a single SE-0482 `staticLibrary` artifactbundle that ships Apple and Linux slices in one bundle, so consumers add a single dependency without platform-conditional manifest logic.

```swift
dependencies: [
    .package(url: "https://github.com/DePasqualeOrg/swift-tokenizers.git", from: "0.5.0")
]
```

### Linux notes

- Linux binaries are built against glibc on Ubuntu 22.04, so they are forward-compatible with the glibc shipped in newer Ubuntu LTS images. musl is not supported.
- The package's PR CI runs `swift test` on Linux against the offline test subset only. Tests that go through `swift-hf-api`'s `HubClient` are gated to macOS because Swift 6.2.x and 6.3.x on Linux crash deterministically in `_HTTPURLProtocol.configureEasyHandle` (`libcurl.Easy Code=43`) on Hugging Face's redirect path; the wrapper itself does not use `URLSession`. The crash is fixed upstream by [swift-corelibs-foundation#5448](https://github.com/swiftlang/swift-corelibs-foundation/pull/5448) and will reach end users in Swift 6.4. See [`docs/linux.md`](docs/linux.md) for more detail.
- Older Xcode versions (and Swift toolchains older than 6.2) cannot resolve the package, since the artifactbundle uses the SE-0482 `staticLibrary` artifact type that landed in Swift 6.2.

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
let tokenIds = try tokenizer.encode(text: "The quick brown fox")
let text = try tokenizer.decode(tokenIds: tokenIds)
```

Use `encodeWithMetadata` when you need the richer encoding data, including token strings, masks, sequence indices, word indices, and offset spans:

```swift
let encoding = try tokenizer.encodeWithMetadata(text: "The quick brown fox")
let tokenIds = encoding.tokenIds
let firstTokenSpan = encoding.offsetSpan(forTokenIndex: 0)
```

`encode`, `decode`, `tokenize`, `encodeBatch`, and the `*WithMetadata` variants throw `TokenizerError` on failure (invalid token IDs, configuration mismatches, internal Rust errors). Wrap calls in `do/catch` when you want to surface specific cases:

```swift
do {
    let text = try tokenizer.decode(tokenIds: tokenIds)
    print(text)
} catch let error {
    switch error {
    case .invalidTokenId(let id):
        print("Token id \(id) is out of range")
    default:
        print(error.localizedDescription)
    }
}
```

### Streaming Detokenizer

For streaming generation, feed tokens one at a time into a `StreamingDetokenizer` that emits text chunks as soon as enough bytes are available to form complete Unicode scalars. Internal state is bounded — the buffer is trimmed after every emission, so it does not grow with stream length.

```swift
let stream = tokenizer.streamingDetokenizer()
for await tokenId in tokenIdStream {
    if let chunk = try stream.consume(tokenId) {
        print(chunk, terminator: "")
    }
    // `nil` means the token's bytes are mid-scalar — wait for the next token.
}
```

When resuming a stream after an interruption, seed the detokenizer with the prior tokens so they are not re-emitted:

```swift
let stream = tokenizer.streamingDetokenizer(initialTokenIds: priorTokens)
```

Streaming uses an uncleaned decode path so retroactive cleanup does not break the byte-prefix invariant. Consumers that want the cleaned form can call `tokenizer.decode(tokenIds:)` on the accumulated IDs at the end of the stream.

### Chat Templates

```swift
let messages: [[String: any Sendable]] = [
    ["role": "user", "content": "Describe the Swift programming language."],
]
let encoded = try tokenizer.applyChatTemplate(messages: messages)
let decoded = try tokenizer.decode(tokenIds: encoded)
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

## Migration

### Throwing API

`Tokenizer.encode`, `decode`, `tokenize`, `encodeBatch`, and the `*WithMetadata` variants now throw `TokenizerError` instead of silently returning empty results on failure. Add `try` to existing call sites:

```swift
// Before
let ids = tokenizer.encode(text: prompt)
let text = tokenizer.decode(tokenIds: ids)

// After
let ids = try tokenizer.encode(text: prompt)
let text = try tokenizer.decode(tokenIds: ids)
```

The `Tokenizer` protocol uses typed throws (`throws(TokenizerError)`), so callers can `catch` exhaustively over the cases without a `default` arm.

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
| Tokenizer load | 386.9 ms | 168.1 ms | 2.3x faster |
| Tokenization | 29.1 ms | 4.5 ms | 6.5x faster |
| Decoding | 35.1 ms | 3.9 ms | 9.0x faster |
| LLM load | 416.9 ms | 183.8 ms | 2.3x faster |
| VLM load | 465.5 ms | 225.8 ms | 2.1x faster |
| Embedding load | 414.3 ms | 197.7 ms | 2.1x faster |

These results were observed on an M3 MacBook Pro using Swift Tokenizers [`0.5.0`](https://github.com/DePasqualeOrg/swift-tokenizers/releases/tag/0.5.0), Swift Transformers [`1.3.2`](https://github.com/huggingface/swift-transformers/releases/tag/1.3.2), and MLX Swift LM [`3.31.3`](https://github.com/ml-explore/mlx-swift-lm/releases/tag/3.31.3).

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

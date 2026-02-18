Swift Tokenizers is a streamlined and optimized fork of Swift Transformers that focuses solely on tokenizer functionality. It has no dependency on the Hugging Face Hub: tokenizers are simply loaded from a directory, and downloading is handled separately.

Tokenizer loading performance is [significantly faster](https://github.com/DePasqualeOrg/swift-tokenizers/pull/3) compared to Swift Transformers, dropping from ~1500 ms to ~300 ms (5x faster) on an M3 MacBook Pro.

## Examples

### Loading a tokenizer

Load a tokenizer from a local directory containing `tokenizer.json` and `tokenizer_config.json`:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(modelDirectory: localDirectory)
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

### Loading tokenizers

Download model files separately, then load from a local directory.

```swift
// Before
let tokenizer = try await AutoTokenizer.from(pretrained: "model-name", hubApi: hub)
let tokenizer = try await AutoTokenizer.from(modelFolder: directory, hubApi: hub)

// After (download tokenizer files to directory first)
let tokenizer = try await AutoTokenizer.from(modelDirectory: directory)
```

# ``Tokenizers``

A high-performance Swift wrapper around Hugging Face's Rust `tokenizers` crate.

## Overview

Swift Tokenizers focuses solely on tokenization and has no dependency on the Hugging Face Hub.

Load a tokenizer from a local directory containing `tokenizer.json` and any sidecar files:

```swift
import Tokenizers

let tokenizer = try await AutoTokenizer.from(directory: localDirectory)
let tokenIds = try tokenizer.encode(text: "The quick brown fox")
let text = try tokenizer.decode(tokenIds: tokenIds)
```

For richer encoding metadata – token strings, masks, sequence indices, word indices, offset spans – use ``Tokenizer/encodeWithMetadata(text:addSpecialTokens:offsetUnit:)``:

```swift
let encoding = try tokenizer.encodeWithMetadata(text: "The quick brown fox")
let firstTokenSpan = encoding.offsetSpan(forTokenIndex: 0)
```

Apply a chat template to format conversation messages into a token sequence:

```swift
let messages: [Message] = [
    ["role": "user", "content": "Describe the Swift programming language."],
]
let encoded = try tokenizer.applyChatTemplate(messages: messages)
```

## Topics

### Loading

- ``AutoTokenizer``

### Tokenizer Protocol

- ``Tokenizer``

### Encoding

- ``TokenizerEncoding``
- ``OffsetSpan``
- ``OffsetUnit``

### Chat Templates

- ``ChatTemplateOverride``
- ``Message``
- ``ToolSpec``

### Errors

- ``TokenizerError``

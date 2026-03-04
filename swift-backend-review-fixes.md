# Swift Backend Review Fixes

Follow-up to `swift-backend-redesign.md`. A code review of the redesign branch surfaced correctness issues concentrated in the Swift backend (every Critical/High finding was Swift-side — the Rust backend came out essentially clean). Since the Swift backend will be kept for now and only retired in a later release, we're fixing all findings with regression tests rather than accepting the divergence.

Every finding below has been verified by reading the upstream source (`/Users/anthony/files/projects/forked/tokenizers/`) and the current Swift code. Where upstream has an equivalent test, we'll port it; otherwise we write a new test, since several of these are also upstream blind spots.

## Summary

| ID | Severity | Area | Fix | Test |
|----|----------|------|-----|------|
| C1 | Critical | WordPiece `max_input_chars_per_word` | Use scalar count | New (upstream gap) |
| C2 | Critical | WordPiece substring walk | Scalar-aligned indexing | New (upstream gap) |
| C3 | Critical | Added-tokens `single_word` boundary | Unicode-aware word chars | Port `single_word_tokens` + Unicode variant |
| C4 | Critical | Swift `tokenize()` stops before post-processor | Run post-processor with `addSpecialTokens: false` to match Python v5 | Parity test across backends |
| H1 | High | Legacy untagged dispatch heuristics | Empty-merges guard + verify order | New fixtures |
| H2 | High | `fuse_unk` leaks into WordPiece/WordLevel | Hardcode `fuseUnknownTokens = false` | New regression test |
| H3 | High | WordLevel silent empty on missing unk | Throw at init | Port `test_tokenize_missing_unk_token` |
| M1 | Medium | `ns.character(at:)` returns UTF-16 units | Read `UnicodeScalar` instead | Covered by C3 test |
| M2 | Medium | NSRegex alternation scaling + ordering | Verify leftmost-longest on overlap | Port `overlapping_tokens` |
| M3 | Medium | bos/eos/unk declared-but-missing silent nil | Document; leave cross-backend consistent | No change |
| M4 | Medium | WordPiece decode separator depends on `decoder == nil` | Add inline assertion comment | Port `pipeline_bert` decoder on/off |
| M5 | Medium | Unigram unknown-fusion divergence | Swift concatenates consecutive unknowns (matches upstream) | Tighten `unigramUnknownCharactersGoToUnk` to `["h","xy","i"]` |
| L1 | Low | `singleWord` Config key reliance on camelCase→snake_case | Add citation comment | No change |
| L2 | Low | `fuseUnknown` pushes raw strings, not unk string | Fine for non-Unigram after H2; cross-check Unigram | Covered by M5 |
| L3 | Low | DeepSeek test comment polish | Add mechanism reference | No change |
| L4 | Low | `.unsupportedModelType("")` vs `<missing>` message | Keep as-is; add test assertion | Existing test |
| L5 | Low | Dead branches in `inferLegacyModelType` | Delete lines 118–120, 124–126 | H1 fixture coverage |
| N1 | Nit | Redundant `as? Int` before `NSNumber.intValue` | Collapse | No change |
| N2 | Nit | Positive: citation comment | Retain | No change |

## Findings

### C1. WordPiece `max_input_chars_per_word` counts graphemes, not scalars

**Location**: `Sources/Tokenizers/WordPieceTokenizer.swift:95`

```swift
if text.count > maxInputCharsPerWord {
    return [unkToken]
}
```

**Upstream**: `tokenizers/src/models/wordpiece/mod.rs:225`
```rust
let char_len = sequence.chars().count();
if char_len > self.max_input_chars_per_word { ... }
```

Swift `String.count` is the grapheme-cluster count. Rust `chars().count()` is the Unicode scalar count. For combining-mark strings (e.g., `"e\u{0301}"` = 1 grapheme / 2 scalars), emoji sequences, and regional-indicator pairs, Swift undercounts relative to upstream. A word Rust would replace with `[UNK]` may pass through on Swift, or vice versa.

**Fix**: `text.unicodeScalars.count`.

**Test**: upstream has no dedicated test. Write two:
1. 101 ASCII characters → `[UNK]` (boundary).
2. 101 `"e\u{0301}"` combining-mark graphemes (= 202 scalars) → `[UNK]` (confirms scalar semantics).
3. 50 `"e\u{0301}"` graphemes (= 100 scalars) → does NOT `[UNK]` (below boundary).

### C2. WordPiece substring walk uses grapheme-cluster indexing

**Location**: `Sources/Tokenizers/WordPieceTokenizer.swift:103–125`

```swift
while start < cursor {
    var candidate = String(text[start..<cursor])
    ...
    cursor = text.index(before: cursor)
}
...
let consumed =
    start > text.startIndex
    ? matched.count - continuingSubwordPrefix.count
    : matched.count
start = text.index(start, offsetBy: consumed)
```

`text.index(before:)` and `text.index(offsetBy:)` advance by grapheme clusters. Arithmetic on `matched.count - continuingSubwordPrefix.count` also counts graphemes.

**Upstream**: `tokenizers/src/models/wordpiece/mod.rs:241–269`
```rust
while start < end {
    let mut substr: Cow<str> = Cow::Borrowed(&sequence[start..end]);
    if start > 0 { substr = Cow::Owned(format!("{}{}", ..., substr)); }
    if self.vocab.contains_key(substr.as_ref()) { ... break; }
    end -= substr.chars().last().map_or(1, |c| c.len_utf8());
}
...
start = end;
```

Upstream walks byte offsets, decrements by the last scalar's UTF-8 length, and advances `start` to `end` directly (byte offset, never a grapheme boundary).

On any input where a grapheme cluster spans multiple scalars AND the vocabulary happens to contain an entry that matches only part of the cluster, Swift will fail to find the match while Rust will find it.

**Fix**: rewrite `tokenize(text:)` to walk `text.unicodeScalars` with `String.UnicodeScalarView.Index` throughout. Advance `start` to the matched `end` directly (as upstream does), no arithmetic on `matched.count`.

**Test**: upstream has no test for this. Write a fixture where a vocabulary entry matches part of a multi-scalar grapheme (e.g., vocab has `e` and `##\u{0301}`, input `"e\u{0301}"`). Swift's current code fails to segment; the fix must produce `["e", "##\u{0301}"]`.

### C3. Added-tokens `single_word` uses ASCII-only word-character check

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:227–236, 259–267`

```swift
/// ASCII word-character check (`[_0-9A-Za-z]`), matching the `\w` semantics of the
/// `regex` crate used in upstream `tokenizers/src/tokenizer/added_vocabulary.rs`.
private static func isAsciiWordChar(_ unit: unichar) -> Bool {
    switch unit {
    case 0x30...0x39, 0x41...0x5A, 0x61...0x7A, 0x5F: return true
    default: return false
    }
}
```

**The claim in the doc comment is wrong.** The Rust `regex` crate's default `\w` is Unicode-aware (`\p{word}`), not ASCII. Upstream uses `Regex::new(r"^\w").unwrap()` and `Regex::new(r"\w$").unwrap()` (`added_vocabulary.rs:99-100`) — both Unicode.

Concrete divergence: added token `<foo>` with `single_word: true` against input `café<foo>`:
- **Rust**: `é` is a Unicode word char → left boundary is a word char → match rejected.
- **Swift**: `é`'s UTF-16 unit is not in the ASCII ranges → match accepted.

**Related (M1)**: `ns.character(at:)` returns a UTF-16 code unit. For supplementary-plane characters (emoji, etc.) it returns a surrogate half, which also isn't in the ASCII ranges. Fixing C3 naturally fixes M1 by switching to scalar-level reads.

**Fix**: replace with a Unicode-aware check operating on `Unicode.Scalar`:
```swift
private static func isWordChar(_ scalar: Unicode.Scalar) -> Bool {
    CharacterSet.alphanumerics.contains(scalar) || scalar == "_"
}
```
At the call site, read the scalar at `outerStart - 1` / `outerEnd` from `text.unicodeScalars`, not the UTF-16 unit from `NSString`.

Update the doc comment to reflect that we're matching upstream's Unicode-aware `\w`.

**Test**: port upstream's `single_word_tokens` (`tokenizers/tests/added_tokens.rs:87–109`) for the ASCII baseline, plus a Unicode variant:
```
input = "département<foo>", added token = "<foo>" with single_word: true
expected: <foo> is NOT matched (surrounded by word chars)
```

### C4. Swift `tokenize()` stops before post-processor

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:370`

Python `transformers` v5 `TokenizersBackend.tokenize()` calls `_encode_plus(..., add_special_tokens=False).tokens()` — i.e., full pipeline with specials suppressed. The Rust backend matches this (`core/tokenizer_core.rs:58` calls `encode(text, false)`). Swift stops before `postProcess`, so its `tokenize()` output diverges whenever a post-processor affects non-special content (e.g., `RobertaProcessing` adjusting spaces, `ByteLevel` normalizing — edge cases).

**Fix**: run the post-processor with `addSpecialTokens: false` in Swift's `tokenize(text:)`.

```swift
private func tokenizeCore(text: String) -> [String] {
    // existing body of tokenize(text:)
}

package func tokenize(text: String) -> [String] {
    postProcess(tokenizeCore(text: text), addSpecialTokens: false)
}

package func encode(text: String, addSpecialTokens: Bool) -> [Int] {
    postProcess(tokenizeCore(text: text), addSpecialTokens: addSpecialTokens).map {
        model.convertTokenToId($0)!
    }
}
```

This avoids double post-processing in `encode`.

**Test**: parity test — on a model with `TemplateProcessing` (`[CLS] $A [SEP]`), both backends' `tokenize("hi")` return tokens with no `[CLS]` / `[SEP]`.

### H1. Legacy untagged dispatch: BPE guard and dead branches

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:101–131`

```swift
private func inferLegacyModelType(...) -> String? {
    let model = tokenizerData["model"]
    if extractedMerges != nil || model["merges"].array() != nil {
        return "BPE"
    }
    if model["continuing_subword_prefix"].string() != nil
        || model["max_input_chars_per_word"].integer() != nil {
        return "WordPiece"
    }
    if case .unigram = extractedVocab { return "Unigram" }       // dead (see L5)
    if model["vocab"].array() != nil { return "Unigram" }
    if case .bpe = extractedVocab { return "WordLevel" }          // dead (see L5)
    if model["vocab"].dictionary() != nil { return "WordLevel" }
    return nil
}
```

**Issues**:

1. **Dead branches** (**L5**): `inferLegacyModelType` is only called when `tokenizerData["model"]["type"]` is nil. But `extractedVocab` is only set to `.bpe` or `.unigram` when `loadTokenizerArtifacts` sees `modelType == "BPE"` or `"Unigram"` respectively (line 537, 545). So the branches checking `extractedVocab` pattern-match against values that can never be reached from this function. Delete lines 118–120 and 124–126.

2. **Empty `merges: []` edge case**: upstream's serde `untagged` tries BPE first; BPE deserializes successfully even with `merges: []`. Our check `model["merges"].array() != nil` may or may not succeed on an empty array — we should verify `Config.array()` semantics and add an explicit test.

3. **Legacy WordPiece-vs-WordLevel ordering**: upstream comment says "WordPiece must stay before WordLevel since WordLevel is a subset of WordPiece". Our heuristic uses distinguishing fields (`continuing_subword_prefix` / `max_input_chars_per_word`) to split them — reasonable, but worth explicit test coverage.

**Fix**: delete dead branches; verify empty-merges behavior; add test fixtures.

**Test**: upstream has no targeted test (their untagged path is implicit in serde). Write small inline fixtures for:
- Untagged BPE (full merges) → dispatches to BPE.
- Untagged BPE with `merges: []` → dispatches to BPE.
- Untagged WordPiece (no `merges`, has `continuing_subword_prefix`) → dispatches to WordPiece.
- Untagged WordLevel (dict vocab only) → dispatches to WordLevel.

### H2. `fuse_unk` leaks into WordPiece and WordLevel

**Location**: `Sources/Tokenizers/WordPieceTokenizer.swift:81`, `WordLevelTokenizer.swift:77`

```swift
self.fuseUnknownTokens = tokenizerConfig.fuseUnk.boolean(or: false)
```

Upstream's `WordPiece` and `WordLevel` (`tokenizers/src/models/wordpiece/mod.rs`, `wordlevel/mod.rs`) have **no** `fuse_unk` field. It's a `Unigram`-only concept. If a downstream `tokenizer_config.json` sets `fuse_unk: true` for a WordPiece model, Swift will fuse consecutive unk tokens via `SwiftTokenizerBackend.fuseUnknown` (line 353) while Rust will not.

**Fix**: hardcode `fuseUnknownTokens = false` on `WordPieceTokenizer` and `WordLevelTokenizer`. Keep the read on `UnigramTokenizer` (that's correct), and only from `tokenizerData.model.fuseUnk`, not `tokenizerConfig.fuseUnk` — the Unigram flag lives in `tokenizer.json`'s `model.fuse_unk`, not in `tokenizer_config.json`.

**Test**: upstream has no test. Write one:
1. Build a WordPiece tokenizer with `tokenizer_config.json` containing `fuse_unk: true`, input `"foo xxx xxx bar"` (where xxx is unk) — assert two separate `[UNK]` tokens, not one fused.

### H3. WordLevel / WordPiece / Unigram fail silently on unresolvable unk

**Location**: `Sources/Tokenizers/WordLevelTokenizer.swift:88–96`, `WordPieceTokenizer.swift:73–74`, `UnigramTokenizer.swift:211–213`.

**Upstream**: `tokenizers/src/models/wordlevel/mod.rs:160–178` and `wordpiece/mod.rs:225–233, 285–293` both return typed `MissingUnkToken` errors. Unigram returns `UnigramError::MissingUnkId`.

**Swift current**: each silently resolves to `nil` and crashes later at `encode`'s force-unwrap `convertTokenToId($0)!` when an unknown input appears.

**Fix**: add new error case `TokenizerError.missingUnknownToken(model: String)`. Throw at init from all three model types when the configured unk can't be resolved to a vocab ID:

```swift
// WordPieceTokenizer.init, after computing tokensToIds and self.unkToken:
guard let unknownTokenId = tokensToIds[unkToken] else {
    throw TokenizerError.missingUnknownToken(model: "WordPiece")
}
self.unknownTokenId = unknownTokenId

// WordLevelTokenizer.init: same pattern, model: "WordLevel"

// UnigramTokenizer.init: validate unk_id is in range before vocab[unknownTokenId] access
guard unknownTokenId < vocabArray.count else {
    throw TokenizerError.missingUnknownToken(model: "Unigram")
}
```

Production configs always have a valid unk; this turns silent runtime crashes into clear init-time errors.

**Test**: port upstream's `test_tokenize_missing_unk_token` (`tokenizers/src/models/wordlevel/mod.rs:241–250`) and extend to cover WordPiece and Unigram:
- WordLevel vocab `{"a": 0, "b": 1}` no unk → init throws.
- WordLevel vocab `{"a": 0, "<unk>": 1}` → init succeeds.
- WordPiece vocab without `[UNK]` → init throws.
- Unigram `unk_id` past vocab length → init throws.

### M2. Verify leftmost-longest on overlapping added tokens

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:203, 216–224`

Pre-sort by length descending + NSRegex leftmost-first alternation gives leftmost-longest for simple containment cases. But the scaling (O(n_tokens × n_matches)) diverges from upstream's Aho-Corasick, and complex overlaps (e.g., `danc`+`nci`+`ing` on `"dancing"`) may not behave identically.

**Fix**: keep NSRegex — rewriting to a trie is out of scope. Ensure correctness via a ported test.

**Test**: port upstream's `overlapping_tokens` (`tokenizers/tests/added_tokens.rs:112–155`). Input `"I like dancing"` with added tokens `danc`, `nci`, `ing` → expect `["I", "Ġlike", "Ġ", "danc", "ing"]` on a byte-level tokenizer. Also port the insertion-order-independence variant at line 130.

### M4. WordPiece decoder-present-or-absent separator

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:426–432`

```swift
let separator = decoder == nil ? " " : ""
let decoded = decodeTokens(tokenStrings).joined(separator: separator)
```

Current code is correct against upstream. The concern is future-proofing: if `DecoderFactory` grows a fallback, this branch silently changes semantics.

**Fix**: add an explaining comment pinning the invariant, and ensure the test locks behavior.

**Test**: port the decoder-on/off assertions from `tokenizers/tests/documentation.rs:494–525` (BERT `pipeline_bert`):
- Same IDs decoded with no decoder → `"welcome to the tok ##eni ##zer ##s library ."`.
- Same IDs decoded with WordPiece decoder → `"welcome to the tokenizers library."`.

### M5. Unigram unknown-fusion: Swift drops; upstream concatenates

**Location**: `Sources/Tokenizers/SwiftTokenizerBackend.swift:353–368` (backend `fuseUnknown`), test at `Tests/TokenizersTests/AlgorithmTests.swift:317–335`.

**Upstream**: `tokenizers/src/models/unigram/model.rs:320–371` concatenates the raw text spans of consecutive unk nodes into a single string token. For input `"hxyi"` with vocab `{<unk>, h, i}`, upstream produces `["h", "xy", "i"]`.

**Swift current**: backend `fuseUnknown` keeps only the first consecutive unk and drops the rest → `["h", "x", "i"]`.

**Fix**: change backend `fuseUnknown` from keep-first to concatenate:

```swift
private func fuseUnknown(_ tokens: [String]) -> [String] {
    guard fuseUnknownTokens else { return tokens }
    var fused: [String] = []
    var previousIsUnknown = false
    for token in tokens {
        let isUnknown = model.convertTokenToId(token) == model.unknownTokenId
        if isUnknown && previousIsUnknown, let last = fused.last {
            fused[fused.count - 1] = last + token
        } else {
            fused.append(token)
        }
        previousIsUnknown = isUnknown
    }
    return fused
}
```

Each Unigram unk node's token string is the raw character at that position, so concatenating token strings matches upstream's raw-byte-span concatenation exactly.

**Test**: tighten `unigramUnknownCharactersGoToUnk` to `tokens == ["h", "xy", "i"]` and remove the Swift/Rust divergence comment.

### M3. Declared-but-missing bos/eos/unk silently resolves to `nil`

**Location**: `WordPieceTokenizer.swift:76–79`, `WordLevelTokenizer.swift:72–75`, also the Rust `resolve_token_id` (cross-backend symmetric).

This is cross-backend-consistent silent-fallback behavior. It's not a divergence from upstream; it's a soft API choice.

**Fix**: leave as-is. Acknowledge in a comment that this matches upstream and the Rust side.

**Test**: no new test.

### L1. `AddedTokenInfo.singleWord` relies on Config camelCase→snake_case

**Location**: `SwiftTokenizerBackend.swift:198`

`addedToken["singleWord"]` works because `Config`'s subscript converts camelCase to snake_case to read `single_word` from JSON. Functional but can confuse future readers.

**Fix**: add a one-line comment citing the Config behavior.

**Test**: none needed.

### L2. `fuseUnknown` pushes the raw token, not the unk token string

**Location**: `SwiftTokenizerBackend.swift:353–368`

`fuseUnknown` appends `token` (the raw string that converts to `unknownTokenId`). After H2, only Unigram fuses, and Unigram's upstream behavior fuses raw text spans — matching ours conceptually. Verify as part of M5.

**Fix**: none after H2 and M5 are addressed.

**Test**: covered by M5.

### L3. DeepSeek test comment could cite mechanism

**Location**: `Tests/TokenizersTests/TokenizerTests.swift:333–344`

Polish: add "Confirmed against Python transformers v5 output for `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`" so a future reader doesn't re-trace §7 of the planning doc.

**Fix**: one-line comment addition.

### L4. `.unsupportedModelType("")` payload is empty but message shows `<missing>`

**Location**: `Sources/Tokenizers/Tokenizer.swift:73–76`, tested in `AlgorithmTests.missingModelTypeThrows`.

Empty-string payload → `errorDescription` renders `<missing>`. If users want to distinguish "empty" from "missing" they have no signal, but this is a low-priority API shape concern.

**Fix**: leave as-is. The test already pins the contract.

**Test**: existing.

### L5. Dead branches in `inferLegacyModelType`

See H1 for discussion. Fix: delete lines 118–120 and 124–126.

### N1. Redundant `as? Int` before `NSNumber.intValue`

**Location**: `WordPieceTokenizer.swift:43–47`, `WordLevelTokenizer.swift:37–41`

```swift
if let id = value as? Int {
    tokensToIds[token] = id
} else if let id = (value as? NSNumber)?.intValue {
    tokensToIds[token] = id
}
```

NSNumber bridges to Int; `as? Int` handles an NSNumber-backed integer. The second branch is redundant for most inputs but defends against weirder NSNumber cases (booleans, unsigned overflow). Low priority.

**Fix**: collapse to a single branch using `(value as? NSNumber)?.intValue`.

### N2. Positive: `AddedTokenInfo.normalized` default citation

**Location**: `SwiftTokenizerBackend.swift:199–201`

The inline comment "Upstream default: `normalized` is `!special` when not explicitly set" correctly cites `added_vocabulary.rs:36–42`. Keep.

## Upstream tests to port

Direct translations of Rust tests where behavior is equivalent:

| Upstream | Path | Purpose |
|----------|------|---------|
| `single_word_tokens` | `tokenizers/tests/added_tokens.rs:87–109` | C3 ASCII baseline |
| `lstrip_tokens` | `tokenizers/tests/added_tokens.rs:36–53` | Added-token flag coverage |
| `rstrip_tokens` | `tokenizers/tests/added_tokens.rs:56–84` | Added-token flag coverage |
| `overlapping_tokens` | `tokenizers/tests/added_tokens.rs:112–155` | M2 |
| `overlapping_tokens` (order-independence) | `tokenizers/tests/added_tokens.rs:130–155` | M2 bonus |
| `test_tokenize_missing_unk_token` | `tokenizers/src/models/wordlevel/mod.rs:241–250` | H3 |
| `pipeline_bert` decoder on/off | `tokenizers/tests/documentation.rs:494–525` | M4 |
| `test_lstrip_unicode_space` | `tokenizers/src/tokenizer/added_vocabulary.rs:1005` | lstrip Unicode space coverage |

Gaps where we write new tests because upstream also lacks coverage:

- **C1** WordPiece `max_input_chars_per_word` boundary (ASCII + Unicode)
- **C2** WordPiece scalar-aligned walk (partial-grapheme vocab entry)
- **C3** `single_word` Unicode boundary (`département<foo>`)
- **H1** Legacy untagged dispatch fixtures (BPE, BPE empty merges, WordPiece, WordLevel)
- **H2** `fuse_unk` on WordPiece — must not fuse
- **M5** Unigram unknown-char fusion (replacing divergent test)
- **C4** Rust `tokenize()` post-processor semantics (parity check)

## Implementation checklist

Commit grouping is designed so each commit builds, tests, and is self-contained.

### Phase 1: Test infrastructure and upstream ports

- [ ] Port `single_word_tokens` from upstream (baseline ASCII; will currently pass)
- [ ] Port `lstrip_tokens`, `rstrip_tokens`
- [ ] Port `overlapping_tokens` (both variants)
- [ ] Port `pipeline_bert` decoder on/off assertions
- [ ] Port `test_lstrip_unicode_space`

*Commit: "Port upstream added-token and decoder regression tests"*

### Phase 2: WordPiece Unicode correctness (C1, C2)

- [ ] Add failing test: WordPiece 101-scalar combining-mark input → `[UNK]`
- [ ] Add failing test: WordPiece partial-grapheme vocab match (`e` + `##\u{0301}`)
- [ ] Switch `max_input_chars_per_word` check to `text.unicodeScalars.count`
- [ ] Rewrite `tokenize(text:)` to walk `UnicodeScalarView` with scalar-aligned indices
- [ ] Verify all previously-passing tests still pass
- [ ] Run Swift-backend trait tests

*Commit: "Fix WordPiece scalar handling (C1, C2)"*

### Phase 3: Added-tokens Unicode correctness (C3, M1)

- [ ] Add failing test: `single_word: true` match rejection with non-ASCII boundary
- [ ] Replace `isAsciiWordChar(_: unichar)` with `isWordChar(_: Unicode.Scalar)` using `CharacterSet.alphanumerics`
- [ ] Update call sites in `findAddedTokenMatches` to read scalars from `text.unicodeScalars`, not UTF-16 units via NSString
- [ ] Update the misleading doc comment ("matching the `\w` semantics of the regex crate" → now actually true)
- [ ] Run full Swift-backend trait tests

*Commit: "Fix added-tokens single_word Unicode boundary (C3, M1)"*

### Phase 4: `fuse_unk` leak (H2)

- [ ] Add failing test: WordPiece with `fuse_unk: true` in tokenizer_config does NOT fuse consecutive unks
- [ ] Hardcode `fuseUnknownTokens = false` in `WordPieceTokenizer` and `WordLevelTokenizer`
- [ ] Verify Unigram still reads its flag correctly (from `tokenizer.json` `model.fuse_unk`, not `tokenizer_config.json`)
- [ ] If Unigram currently reads from `tokenizerConfig.fuseUnk`, move to `tokenizerData.model.fuseUnk`

*Commit: "Restrict fuse_unk to Unigram (H2)"*

### Phase 5: Unresolvable-unk hard-fail (H3)

- [ ] Add `TokenizerError.missingUnknownToken(model: String)` case + `errorDescription`
- [ ] Add failing tests:
  - [ ] WordLevel init with vocab `{"a": 0, "b": 1}` (no unk) throws
  - [ ] WordPiece init with vocab lacking `[UNK]` throws
  - [ ] Unigram init with `unk_id` past vocab length throws
- [ ] Add passing tests:
  - [ ] WordLevel with `<unk>` in vocab succeeds; unknown input → `["<unk>"]`
  - [ ] WordPiece with `[UNK]` in vocab succeeds
- [ ] Throw in `WordLevelTokenizer.init`, `WordPieceTokenizer.init`, `UnigramTokenizer.init` (both designated inits) when unk can't be resolved

*Commit: "Fail fast on unresolvable unk token across model types (H3)"*

### Phase 6: Legacy dispatch cleanup (H1, L5)

- [ ] Add fixtures for untagged BPE (full merges), untagged BPE (empty merges), untagged WordPiece, untagged WordLevel
- [ ] Add tests asserting each fixture dispatches to the expected model type
- [ ] Delete dead branches `inferLegacyModelType` lines 118–120 and 124–126
- [ ] Verify `Config.array()` on `[]` behavior; adjust the merges guard if needed

*Commit: "Verify legacy untagged dispatch coverage and remove dead branches (H1, L5)"*

### Phase 7: Unigram unknown-fusion parity (M5, L2)

- [ ] Change `SwiftTokenizerBackend.fuseUnknown` from keep-first to concatenate-consecutive
- [ ] Tighten `unigramUnknownCharactersGoToUnk` to `tokens == ["h", "xy", "i"]` and remove the divergence comment
- [ ] Verify both backends produce identical output on the same fixture

*Commit: "Concatenate consecutive unknowns in Unigram fusion (M5, L2)"*

### Phase 8: Minor polish (M3, M4, L1, L3, L4, N1, N2)

- [ ] M4: add explanatory comment on `decoder == nil` separator branch
- [ ] M3: add comment acknowledging cross-backend silent bos/eos fallback
- [ ] L1: add citation for Config camelCase→snake_case on `addedToken["singleWord"]`
- [ ] L3: add mechanism reference to DeepSeek test comment
- [ ] N1: collapse redundant `as? Int` in both `WordPieceTokenizer.swift` and `WordLevelTokenizer.swift`

*Commit: "Polish comments and minor redundancies (M3, M4, L1, L3, N1)"*

### Phase 9: Swift tokenize() post-processor alignment (C4)

- [ ] Split `SwiftTokenizerBackend.tokenize(text:)` into private `tokenizeCore` + public `tokenize` that calls `postProcess(..., addSpecialTokens: false)`
- [ ] Update `encode` to call `tokenizeCore` directly to avoid double post-processing
- [ ] Add parity test: both backends' `tokenize("hi")` on a `TemplateProcessing` model return identical tokens (no `[CLS]`/`[SEP]`)
- [ ] Run full Swift-backend trait tests; update any test whose expectation was based on the pre-post-processor shape

*Commit: "Run post-processor in Swift tokenize() with specials suppressed (C4)"*

### Phase 10: Final validation

- [ ] Run Swift-backend trait tests
- [ ] Run Rust-backend trait tests (with `TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH` if FFI changed)
- [ ] Run `cargo fmt --check`, `cargo clippy` on Rust crate if changed
- [ ] Cross-check all Critical/High items have regression tests that fail before the fix and pass after

## Open questions — resolved

Investigation details below. The Summary table, per-finding sections, and checklist phases above have been updated with the resolved direction.



### Q1 (C4): tokenize() post-processor semantics

**Investigation**:
- Python `transformers` v5 `TokenizersBackend.tokenize()` (`transformers/src/transformers/tokenization_utils_tokenizers.py:779`):
  ```python
  def tokenize(self, text, pair=None, add_special_tokens=False, **kwargs):
      return self._encode_plus(text=text, text_pair=pair, add_special_tokens=add_special_tokens, **kwargs).tokens()
  ```
  Full pipeline with `add_special_tokens=False` by default.
- Rust backend (`rust/swift-tokenizers-rust/src/core/tokenizer_core.rs:58`):
  ```rust
  pub(crate) fn tokenize(&self, text: &str) -> Result<Vec<String>, CoreError> {
      let encoding = self.tokenizer.encode(text, false)...
  ```
  Matches Python v5 exactly (upstream `Tokenizer::encode(text, false)` runs normalizer → pre-tokenizer → model → post-processor, with specials suppressed).
- Swift backend (`SwiftTokenizerBackend.swift:370`): stops before `postProcess`.

**Resolution**: align Swift `tokenize()` with Python v5 / Rust — run the post-processor with `addSpecialTokens: false`. This is the canonical Python-v5 contract, and already what the Rust backend does.

**Expected blast radius**: low. Existing callers (`TokenizerTests.swift:133, 371, 391–440`, `AlgorithmTests.swift:281, 295, 313, 328, 391, 394`) test against outputs that don't include special tokens. For `TemplateProcessing` models, `addSpecialTokens=false` suppresses `SpecialToken` entries → no change. For `Bert`/`RobertaProcessing`, those also respect `addSpecialTokens=false` → no change. For `ByteLevel` post-processors, no-op. If any test does break, the old assertion was wrong relative to Python.

**Implementation**: introduce a private `tokenizeCore(text:)` that stops before post-processing, used by `encode()`; make public `tokenize(text:)` call `tokenizeCore` then `postProcess(..., addSpecialTokens: false)`. Avoids double post-processing in `encode`.

**Test**: parity assertion — for a model with `TemplateProcessing` (`[CLS] $A [SEP]`), both backends' `tokenize("hi")` must return tokens without `[CLS]` / `[SEP]`.

### Q2 (M5): Unigram unknown-fusion direction

**Investigation**: upstream `tokenizers/src/models/unigram/model.rs:320–333` (optimized) and `354–371` (unoptimized), both cases:
```rust
if self.fuse_unk && Some(node.id) == self.unk_id {
    token.push(sentence[starts_at..ends_at].to_string());
} else {
    if !token.is_empty() {
        token.reverse();
        results.push(token.concat());  // ← concatenates raw text spans
        token = vec![];
    }
    results.push(sentence[starts_at..ends_at].to_string());
}
```

Swift current behavior (`SwiftTokenizerBackend.fuseUnknown` lines 353–368):
```swift
if isUnknown {
    if !previousIsUnknown {
        fused.append(token)  // keeps FIRST, drops the rest
    }
} else { fused.append(token) }
```

Concrete divergence on input `"hxyi"` with vocab `{<unk>, h, i}`:
- Upstream: `["h", "xy", "i"]` — concatenates consecutive unknowns.
- Swift: `["h", "x", "i"]` — keeps first, drops rest.

**Resolution**: align Swift to upstream. Each Unigram unk node produces a token string whose text is the raw character (via `lattice.insert(startOffset: beginPos, length: mblen, ...)`), so concatenating consecutive unk token strings at the backend level matches upstream's raw-text-span concatenation exactly.

**Implementation**: change `SwiftTokenizerBackend.fuseUnknown` to accumulate consecutive unknowns and concatenate them on flush, rather than keeping only the first:
```swift
if isUnknown {
    if previousIsUnknown, let last = fused.last {
        fused[fused.count - 1] = last + token
    } else {
        fused.append(token)
    }
} else { fused.append(token) }
```

**Test**: tighten `unigramUnknownCharactersGoToUnk` from `tokens.first == "h" && tokens.last == "i"` to `tokens == ["h", "xy", "i"]`. Remove the "Swift vs Rust" divergence comment. Verify both backends produce the same output.

### Q3 (H3): Error case for missing unk

**Investigation**: existing `TokenizerError` cases (`Tokenizer.swift:35–45`) have structured error kinds: `missingConfigField(field: String, component: String)`, `unsupportedComponent(kind: String, type: String)`, etc. Upstream uses dedicated typed errors (`Error::MissingUnkToken`, `UnigramError::MissingUnkId`).

The failure mode is distinct from `.missingVocab` (vocab is present, it just lacks the unk) and `.missingConfigField` (field may be configured but not resolved to a vocab ID). Reusing them would blur semantics.

**Resolution**: add `TokenizerError.missingUnknownToken(model: String)` with message `"Configured unknown token is not present in the \(model) vocabulary."`.

**Scope**: same issue exists for `WordPieceTokenizer` — it reads `unkToken ?? "[UNK]"` but silently sets `unknownTokenId = nil` if `"[UNK]"` isn't in vocab, then `encode`'s `convertTokenToId($0)!` crashes on an unknown input. Fix symmetrically: both `WordPieceTokenizer` and `WordLevelTokenizer` throw at init if `unknownTokenId` cannot be resolved. `UnigramTokenizer` already requires `unkId` in config and resolves it from the vocab array; if the index is out of bounds, that crashes today — also fix by throwing `.missingUnknownToken(model: "Unigram")`.

**Test**: (a) fixture with vocab that lacks `[UNK]` and no override → `WordPieceTokenizer.init` throws. (b) vocab with `[UNK]` → succeeds. (c) same matrix for WordLevel (`<unk>` default). (d) Unigram with `unk_id` pointing past vocab length → throws.

## Updated implementation notes

The resolutions above tighten a few existing checklist items:

- **Phase 5 (H3)**: error case is now concrete (`.missingUnknownToken(model: String)`). Extend the fix to `WordPieceTokenizer` and `UnigramTokenizer` as well, not just `WordLevelTokenizer`.
- **Phase 7 (M5)**: direction is now concrete (concatenate, not keep-first). The fix is ~5 lines in `fuseUnknown`.
- **Phase 9 (C4)**: fix is now concrete (split `tokenize` into public + `tokenizeCore`, public runs `postProcess(..., addSpecialTokens: false)`). No FFI change needed — the Rust side is already correct.

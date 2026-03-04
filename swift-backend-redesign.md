# Swift Backend Redesign: Match Canonical Behavior

Status: draft, pre-implementation. The goal is for both Swift and Rust backends of `swift-tokenizers` to produce the same tokenization as upstream `huggingface/tokenizers` and Python transformers v5's generic `TokenizersBackend` load-from-`tokenizer.json` path, for models with a well-formed `tokenizer.json`, over the single-string, non-padding/truncation API surface that swift-tokenizers currently exposes. (This is deliberately narrower than "every `AutoTokenizer.from_pretrained` path in Python v5" — v5 still routes some models through model-specific subclasses with per-class constructors; see §1.2. We align with the class-agnostic backend path, not the per-class legacy.) Python v5 additionally honors `truncation` and `padding` blocks from `tokenizer.json` at `tokenization_utils_tokenizers.py:394-411`; that's out of scope here — see §9. This is an ideal-state design, not a staged migration.

## 1. Canonical behavior

For a model loaded via `AutoTokenizer.from_pretrained(dir)` where `tokenizer.json` is present:

- **Upstream `huggingface/tokenizers` (Rust crate):** loads `tokenizer.json` verbatim via `Tokenizer::from_bytes`. Dispatches on `model.type` (`tokenizers/src/models/mod.rs:62-69`: `BPE` / `WordPiece` / `WordLevel` / `Unigram`). No sidecar consulted.
- **Python transformers v5:** `tokenization_utils_base.py:1783-1785` pops `add_bos_token` and `add_eos_token` from `init_kwargs` before the tokenizer class `__init__` runs. Inside `TokenizersBackend.__init__` (`tokenization_utils_tokenizers.py:422`) `_should_update_post_processor = explicit_bos_eos_in_kwargs or self._tokenizer.post_processor is None`. Both conditions are false for a well-formed config, so the base-class `update_post_processor` (line 522) is never called. The generic `TokenizersBackend.__init__` does not call `_post_init` either. Some model-specific subclasses define or call post-init hooks from their own constructors, and those can fire when Python selects those subclasses. This redesign intentionally excludes those per-class constructor compatibility paths by dropping class-name dispatch entirely (§3.1).

**Net effect, for a well-formed `tokenizer.json`:** the post-processor, normalizer, pre-tokenizer, and decoder encoded in `tokenizer.json` are used as-is. Fields in `tokenizer_config.json` such as `add_bos_token` and `add_eos_token` are advisory only — they are not applied to tokenization.

This redesign makes both `swift-tokenizers` backends follow the same rule.

### 1.1 Python v5's direction of travel

This redesign aligns with the direction Python transformers v5 is actively moving, even though v5 still retains per-class dispatch for other reasons (see §1.2):

- **`MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`** (`tokenization_auto.py:346-380`) — a set of 31 `model_type` values, including `deepseek_v2`, `deepseek_v3`, `internlm2`, `phi3`, `modernbert`, `cohere_asr`, etc. The comment above the set calls them "Models with incorrect `tokenizer_class` in their Hub `tokenizer_config.json` files" that "will be forced to use `TokenizersBackend`." Mechanically, the force-map overrides `TOKENIZER_MAPPING_NAMES[model_type]` to `"TokenizersBackend"` (line 382-384); the actual routing effect depends on what else the repo ships. When `tokenizer_class` is absent from `tokenizer_config.json`, this directly selects `TokenizersBackend` at the fallback branch (around line 821). When `tokenizer_class` is present (e.g., `"LlamaTokenizerFast"`), routing depends on whether the repo also ships an `auto_map`: without it, the class-mismatch branch at `709-727` fires and selects the concrete class (e.g., `LlamaTokenizer`); with `auto_map`, lines `759-762` disable remote code for the force-map set, then the `tokenizer_class` fallback at `788-799` resolves the same concrete class. §7 walks the full trace for the DeepSeek case. Either way, the set encodes v5's stance that this class of malformed-metadata repo is a dispatch-policy problem rather than a per-class compatibility problem, which is the same stance this redesign takes (§3.1).
- **Many new model types map directly to `TokenizersBackend`** in `TOKENIZER_MAPPING_NAMES` from day one (`("glm", ...)`, `("glm4", ...)`, `("got_ocr2", ...)`, `("olmo3", ...)`, and ~20 others).
- **The `Fast`/slow distinction is retired.** All fast tokenizer classes drop the `Fast` suffix; `LlamaTokenizerFast = LlamaTokenizer` is a backward-compat alias (`models/llama/tokenization_llama.py:149`).

### 1.2 Why swift-tokenizers can drop per-class dispatch entirely

Python v5 still keeps per-class tokenizer classes (`LlamaTokenizer`, `GemmaTokenizer`, `Qwen2Tokenizer`, ~45 total) even though the load-from-`tokenizer.json` path is class-agnostic. Two reasons:

1. **Backward compatibility.** User code in the wild says `LlamaTokenizer.from_pretrained(...)` or imports the class name directly. Removing those classes would break user code.
2. **Construct-from-scratch.** Python supports building a tokenizer from vocab+merges alone (no `tokenizer.json`) — e.g., `LlamaTokenizer(vocab={...}, merges=[...])`. This path needs per-class knowledge of how to assemble the pipeline (which normalizer, pre-tokenizer, decoder, etc.). `TokenizersBackend` alone cannot handle it.

swift-tokenizers has neither concern in a load-bearing way. Among the model types, only `BertTokenizer` is `public` today; `BPETokenizer` and `UnigramTokenizer` are already `internal`. `BertTokenizer` direct construction from vocab+merges is reachable, but that surface is undocumented, not referenced in the README, and has no external consumers (the only in-repo user of the `knownTokenizers` registry is a concurrency-stress test). The redesign deliberately removes this undocumented direct-construction surface: `AutoTokenizer.from(directory: URL)`, which requires `tokenizer.json`, becomes the sole documented entry point, and the new `WordPieceTokenizer` / `WordLevelTokenizer` are introduced as `internal` to match the other model types (§4). So we can structurally be simpler than Python v5: drop per-class dispatch, drop the registry, drop the class-name allow-list, and still arrive at the same canonical output for well-formed configs.

## 2. What's wrong today

### 2.1 Swift backend has non-canonical plumbing

- Dispatches on `tokenizer_class` (a Python-side wrapper name) via a 20-entry `knownTokenizers` registry in `SwiftTokenizerBackend.swift:44-66`. The registry collapses to 3 real algorithms (`BertTokenizer`, `BPETokenizer`, `UnigramTokenizer`) plus one decorative alias (`T5Tokenizer: UnigramTokenizer {}` at line 683, zero overrides).
- Falls back to `BPETokenizer.self` for unknown class names (`SwiftTokenizerBackend.swift:87`) — silently wrong for WordPiece or Unigram models. "Strict mode" converts this to a loud error via a per-name allow-list (`TokenizerCompatibility.rustSupportedTokenizerNames`).
- Applies Llama-specific extras when `tokenizer_class == "LlamaTokenizer"`: post-processor rebuild (`LlamaTokenizerConfig.updatedPostProcessorConfig`), pre-tokenizer swap to `Metaspace` with `normalizer` stripped (`LlamaTokenizerConfig.buildUpdatedConfig`, non-legacy branch), and a `tokenize()` override for SentencePiece underline handling (`LlamaSwiftTokenizerBackend`). None of this is canonical v5 behavior for well-formed configs.

### 2.2 Rust backend has non-canonical plumbing

- `rust/swift-tokenizers-rust/src/core/tokenizer_core.rs:69-111` applies a `LlamaTokenizer`-name-gated post-processor shim that prepends/appends bos/eos token IDs on top of `tokenizer.encode(...)` output (lines 131-144). This mirrors pre-v5 `LlamaTokenizerFast.update_post_processor` semantics, not v5's.
- Swift-side `RustAutoTokenizerDirectoryLoader.validateStrictCompatibility` (`RustBackedTokenizer.swift:367-381`) applies the same Swift-backend allow-list to the Rust path, rejecting models the Rust crate can tokenize correctly.

### 2.3 Tests encode non-canonical expectations

One test (`deepSeekPostProcessor`) explicitly comments that "Deepseek needs a post-processor override to add a bos token as in the reference implementation" — admitting it's an override. Verified directly: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/tokenizer.json` has `post_processor.type = "ByteLevel"`, which does not add bos/eos at the token-ID level. Canonical Python v5 and upstream Rust would both produce `[15191, 525, 498, 30]` for `"Who are you?"`. The current test expects `[151646, 15191, 525, 498, 30]` — only achievable via our pre-v5 shim.

All other Llama-family tests (`llamaPostProcessor`, `legacyLlamaBehaviour`, `FactoryTests.llama`) use models whose `tokenizer.json` already ships a working `TemplateProcessing` post-processor (verified: Phi-3 and Llama-2 both have `TemplateProcessing`). They pass under canonical behavior unchanged.

## 3. Target design

### 3.1 Dispatch

Swift dispatches on `tokenizer.json`'s `model.type`:

```
model.type == "BPE"       → BPETokenizer
model.type == "WordPiece" → WordPieceTokenizer
model.type == "Unigram"   → UnigramTokenizer
model.type == "WordLevel" → WordLevelTokenizer
```

An unrecognized value throws `TokenizerError.unsupportedModelType(String)`. No silent fallback. A missing `model.type` field throws the same error (with an empty or nil string indicated in the message).

**Deliberate divergence from upstream Rust:** upstream supports a legacy untagged deserialization path at `tokenizers/src/models/mod.rs:125-133` that infers the algorithm from the shape of the `model` object when `type` is absent. Swift does not replicate this — throwing on missing is a tighter invariant. Untagged configurations are extremely rare in practice (modern `tokenizers` releases always emit `type` when serializing), so the cost of strictness is low. If real-world untagged configs surface, revisit.

Rust backend dispatch is unchanged — the upstream crate already dispatches on `model.type` (and retains its legacy untagged fallback, which Swift does not).

### 3.2 No post-processor shim

Both backends delete their Llama compatibility shims entirely. Rationale:

- Canonical Python v5 skips the shim for well-formed configs.
- Upstream Rust tokenizers never applied it.
- Our pre-v5 shim only fires usefully when `tokenizer.json` is malformed (lacks a proper `post_processor`). For those models, canonical Python and upstream Rust also produce "incorrect" output. We are not more correct than the upstream libraries.

Concretely:

- **Swift:** delete `LlamaSwiftTokenizerBackend`, `LlamaTokenizerConfig`, and `SwiftAutoTokenizerFactory.makeLlamaTokenizer`. All paths go through a single factory that dispatches on `model.type`.
- **Rust:** delete `llama_post_processor_compatibility` and `LlamaPostProcessorCompatibility` in `tokenizer_core.rs`. Delete the on-top token-ID injection in `encode` (lines 131-144). Delete `normalized_tokenizer_name`.

### 3.3 Strict mode and registry

Remove both:

- Public API: drop the `strict: Bool = true` parameter from `AutoTokenizer.from(directory:)`.
- Swift: delete `RegisteredTokenizerStore`, `TokenizerModel.registerTokenizer`, `TokenizerModel.tokenizerClass(for:)`, the `knownTokenizers` dictionary, and the `public extension AutoTokenizer { static func register... }` block.
- Rust side: delete `RustAutoTokenizerDirectoryLoader.validateStrictCompatibility` and associated plumbing.

### 3.4 `TokenizerCompatibility.swift`

Deleted entirely. `modelTypeToTokenizerClass`, `rustSupportedTokenizerNames`, `resolvedTokenizerClass`, `resolvedTokenizerName`, `validateResolvedTokenizerName`, `isRustSupportedTokenizer`, and `normalizedTokenizerName` are all unreferenced after the shim deletions.

### 3.5 `TokenizerRuntimeConfiguration`

Remove these fields (unused after shim deletion):

- `tokenizerClass: String?`
- `modelType: String?`
- `addBosToken: Bool`
- `addEosToken: Bool`
- `legacy: Bool?`

Rust-side `RuntimeConfiguration` struct (`rust/swift-tokenizers-rust/src/core/sidecars.rs`) gets the same treatment. FFI JSON schema changes accordingly — single atomic change across both sides.

`bosToken`, `eosToken`, `unknownToken`, `sepToken`, `padToken`, `clsToken`, `maskToken`, `additionalSpecialTokens`, `cleanUpTokenizationSpaces`, `modelMaxLength`, `chatTemplate`, `fuseUnknownTokens` all stay — they're used by chat-template context building (`chatTemplateContextObject`) and/or the tokenization pipeline.

### 3.6 Sidecar loaders

- Swift `AutoTokenizerDirectorySidecars.load`: remove the `config.json` merge that backfilled `tokenizer_class` — no longer needed.
- Rust `sidecars.rs`: remove the `config.json` read fallbacks for `tokenizer_class` / `model_type`. If `config.json` is no longer read for anything, remove that code path entirely.
- Chat-template override loading (`chat_template.jinja`, `chat_template.json`) unchanged on both sides.

### 3.7 New types

- `WordPieceTokenizer` — *rewrite* of `BertTokenizer` as a pure WordPiece model step. Implements only vocab lookup with longest-prefix subword match; reads `unk_token`, `continuing_subword_prefix`, and `max_input_chars_per_word` from `tokenizerData.model`, matching upstream `tokenizers/src/models/wordpiece/mod.rs:32-50`. Does not run its own normalizer or pre-tokenizer — the backend's `Normalizer` / `PreTokenizer` factories (already loaded from `tokenizer.json` at `SwiftTokenizerBackend.swift:198-200`) handle that. Today's `BertTokenizer` runs a hardcoded `BasicTokenizer` plus fixed `[UNK]` / `##` / `max=100` inside the model step, on top of what `SwiftTokenizerBackend` already applies — tolerable today because the BERT normalizer output is roughly idempotent under `BasicTokenizer`, but carrying that double-pipeline forward under the name `WordPieceTokenizer` would misrepresent what the type is. `BasicTokenizer` is deleted along with the rewrite; its standalone public API is not exported by any consumer. Visibility: `internal` (matching `BPETokenizer` / `UnigramTokenizer`, not the old public `BertTokenizer` — see §4).
- `WordLevelTokenizer` — new. Implements vocab lookup with unk fallback. Included not because Python transformers uses it (zero model classes declare `model = WordLevel`), but because upstream Rust supports it and leaving it out of Swift would create a backend asymmetry (some `tokenizer.json` files would load in the Rust backend and fail in Swift). Visibility: `internal`, matching the other model types.

### 3.8 End-state summary

| Concern | Swift (today) | Swift (target) | Rust (today) | Rust (target) |
|---------|--------------|----------------|--------------|---------------|
| Algorithm dispatch | `tokenizer_class` → class | `model.type` → class | upstream crate (`model.type`) | unchanged |
| Unknown algorithm | Silent BPE fallback + strict gate | Loud `unsupportedModelType` | Upstream crate error | unchanged |
| Registry / `AutoTokenizer.register` | Yes | Removed | N/A | N/A |
| Strict mode | Yes | Removed | Yes (allow-list) | Removed |
| Llama post-processor shim | Name-gated, rebuilds | Removed | Name-gated, on-top injection | Removed |
| Llama Metaspace pre-tokenizer swap | Yes | Removed | No | unchanged (still no) |
| Llama `tokenize()` underline override | Yes | Removed | No | unchanged |
| `tokenizer_class` / `model_type` in `RuntimeConfiguration` | Present | Removed | Present | Removed |
| `add_bos_token` / `add_eos_token` / `legacy` in `RuntimeConfiguration` | Present | Removed | Present | Removed |

Both backends collapse to "trust `tokenizer.json`, dispatch on `model.type`." This eliminates the two major sources of divergence — class-name dispatch and name-gated Llama shims. The Rust backend achieves close-to-upstream parity by delegation (it *is* the upstream crate); the Swift backend still reimplements every component locally (`Normalizer`, `PreTokenizer`, `PostProcessor`, `Decoder`, model types), so parity there remains a test-backed property (§6), not an architectural guarantee. Targeted component and integration tests cover the remaining divergence surface.

## 4. Public API impact

Breaking changes, no source-compat shims:

- `AutoTokenizer.from(directory:strict:)` → `AutoTokenizer.from(directory:)` — `strict` parameter removed.
- `AutoTokenizer.register(_:for:)` — removed.
- `BertTokenizer` (previously `public`) — removed. Replaced by `WordPieceTokenizer`, which is `internal` to match `BPETokenizer` / `UnigramTokenizer`. The rationale for making it internal rather than carrying forward `BertTokenizer`'s public status: `AutoTokenizer.from(directory:)` is the sole documented entry point (§1.2), and none of the other model types expose a public type. If a future need for public low-level construction emerges, it can be promoted deliberately.
- `WordLevelTokenizer` — new, `internal` visibility, for the same reason.
- `BasicTokenizer` (previously `public`) — removed.
- `TokenizerError.unsupportedTokenizer(String)` and `.missingTokenizerClassInConfig` — removed.
- `TokenizerError.unsupportedModelType(String)` — new.

Unchanged: `PreTrainedTokenizer`, `Tokenizer`, `TokenizingModel`, `PreTrainedTokenizerModel`, `TokenizerVocab`, `TokenizerMerges`, `ChatTemplateArgument`.

## 5. Resolved decisions

### 5.1 WordPiece rewrite

Move `Sources/Tokenizers/BertTokenizer.swift` → `WordPieceTokenizer.swift` via `git mv`, then rewrite the contents. Replace the three types (`BertTokenizer`, `BasicTokenizer`, `WordpieceTokenizer`) with a single `WordPieceTokenizer` that conforms to `PreTrainedTokenizerModel` and implements only the WordPiece model step — longest-prefix subword match over the vocab, using `unk_token` / `continuing_subword_prefix` / `max_input_chars_per_word` read from `tokenizerData.model` (defaults `"[UNK]"` / `"##"` / `100`, matching upstream Rust). No internal normalization, no internal pre-tokenization — the backend pipeline applies those from `tokenizer.json`. `BasicTokenizer` is deleted.

Move `Tests/TokenizersTests/BertTokenizerTests.swift` → `WordPieceTokenizerTests.swift` via `git mv`. Tests that directly constructed `BertTokenizer(vocab:merges:...)` or exercised `BasicTokenizer` either get rewritten to go through `AutoTokenizer.from(directory:)` against a WordPiece fixture, or are deleted if they duplicate coverage already in `TokenizerTests.bertUncased` / `bertCased`. The SQuAD-based `fullBertTokenizer` test is worth preserving in rewritten form — it goes through the full backend pipeline against a BERT fixture.

The decorative `private final class T5Tokenizer: UnigramTokenizer {}` alias is deleted.

### 5.2 `WordLevel` support

Add `Sources/Tokenizers/WordLevelTokenizer.swift` (vocab lookup + unk fallback). Register in `Package.swift`'s `tokenizerSwiftBackendSources`. Add a small local fixture + test.

### 5.3 Shim design

No shim on either backend. Canonical behavior for well-formed configs is "trust `tokenizer.json`"; we do the same. Non-canonical Llama extras on both backends get deleted.

### 5.4 Cleanup semantics

`SwiftTokenizerBackend.performsCleanup = true` vs `RustTokenizerBackend.performsCleanup = false` stays as-is. The facade already compensates (`PreTrainedTokenizer.decode` re-applies cleanup if `!performsCleanup`). Out of scope; flag as a follow-up only if implementation surfaces divergence in `cleanUpTokenizationSpaces` output.

## 6. Test plan

### Tests to update

- **`TokenizerTests.deepSeekPostProcessor`**: change the expected IDs from `[151646, 15191, 525, 498, 30]` to `[15191, 525, 498, 30]`. Rewrite the comment: "DeepSeek's `tokenizer.json` has a `ByteLevel` post-processor, which does not add bos/eos. This matches canonical Python transformers v5 and upstream Rust."
- **`TokenizerTests.nllbTokenizer`**: remove the strict-throws assertion and the `strict: false` call; just load and assert a correct encode.
- **`TokenizerTests.concurrentTokenizerRegistration`**: deleted along with `AutoTokenizer.register`.

### Tests expected to pass unchanged

- All other model-behavior tests: `bertUncased`, `bertCased`, `bertCasedResaved`, `robertaEncodeDecode`, `robertaXLMTokenizer`, `robertaXLMCanonicalTokenizer`, `kredorPunctuateAllTokenizer`, `legacyLlamaBehaviour`, `llamaPostProcessor`, `tokenizerBackend`, `localTokenizerFromDownload`, `FactoryTests.llama`, `FactoryTests.whisper`, all of `ChatTemplateTests`.
- `BertTokenizerTests.fullBertTokenizer` is preserved in rewritten form (`WordPieceTokenizerTests.fullBertTokenizer`) — the assertions stay but the setup routes through `AutoTokenizer.from(directory:)` against a BERT fixture rather than constructing `BertTokenizer` directly. Other tests in that file that exercised `BasicTokenizer` or direct `BertTokenizer` construction (`testBasicTokenizer`, `fullBasicTokenizer`, `wordpieceDetokenizer`, `encoderDecoder`, Chinese-tokenization cases) are either rewritten to go through the pipeline or deleted as duplicates of `bertUncased`/`bertCased` coverage (see §5.1).
- Component factory tests (`NormalizerTests`, `PreTokenizerTests`, `PostProcessorTests`, `DecoderTests`) — already dispatch on `tokenizer.json` field types.

### New coverage — required for the redesign

- **WordLevel algorithm test**. Small local fixture; exercise encode/decode against it. Confirms the new `WordLevelTokenizer` (§3.7) works end-to-end. The fixture must either (a) set `clean_up_tokenization_spaces: false` in `tokenizer_config.json`, or (b) avoid test text that triggers cleanup rewrites (` .`, ` ,`, ` '`, etc.) — otherwise the backend's cleanup pass rewrites the decode output and masks the space-join fallback the test is meant to assert.
- **`TokenizerError.unsupportedModelType` test**. Fixture with `model.type = "Nonsense"` in `tokenizer.json`; assert that `AutoTokenizer.from(directory:)` throws `.unsupportedModelType("Nonsense")`. Confirms §3.1's "no silent fallback" rule.
- **Algorithm dispatch test**. One tiny local fixture per `model.type` (BPE, WordPiece, Unigram, WordLevel); for each, `AutoTokenizer.from(directory:)` must return the expected concrete class. Direct regression guard for §3.1 — without this, a dispatch bug only surfaces indirectly as a wrong-tokens diff in an integration test.
- **Swift/Rust parity test**. Same directory loaded on both backends produces identical `tokenize` / `encode` / `decode`. Under this design the backends are expected to agree for supported `tokenizer.json` features (the Rust side delegates to upstream; the Swift side reimplements components locally); this test makes that expectation explicit rather than leaving it to chance. Cheap regression guard covering the whole redesign in one assertion. Promoted from "optional" to required.
- **Added-tokens behavior tests**. `lstrip`, `rstrip`, `single_word`, and `normalized` entries in `tokenizer.json`'s `added_tokens`, analogous to upstream `tokenizers/tests/added_tokens.rs`. Required (not recommended) because these fields are part of `tokenizer.json` — the "trust `tokenizer.json`" claim falls over if they are not honored. Today these behaviors are only exercised implicitly through whole-model outputs.

### New coverage — recommended additions

Gaps that predate the redesign but are worth closing while the test suite is already being touched:

- **TemplateProcessing functional tests**. Only error cases are covered today. The "no shim" stance (§3.2) depends on TemplateProcessing producing correct bos/eos insertion for well-formed configs, so this processor deserves direct functional coverage — single-sequence expansion, pair-sequence expansion, special-token substitution.
- **Unigram algorithm unit tests**. Small local fixtures exercising scoring/segmentation independently of any real model, analogous to `tokenizers/tests/unigram.rs`. Today the Swift suite exercises Unigram only indirectly through Llama/T5 integration.
- **Multi-script roundtrip sample**. One shared test string (emoji, CJK, Arabic, Thai, whitespace-heavy runs) applied to each of the 8 parameterized models in `TokenizerTests.tokenizer`. Mirrors the shared `input_string` pattern in Python transformers v5 tests. Generate expected outputs once from canonical reference implementations and store them in the existing edge-case JSON.
- **Extended hub-integration coverage**. Compact integration tests (no full encoded-tokens fixture files — a few asserted encode outputs per model in the `phi4` / `deepSeekPostProcessor` style) for families that either were previously rejected by strict mode or sit on Python transformers v5's `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` force-map list. Each one directly validates the "dispatch on `model.type`, trust `tokenizer.json`" rule against a class name that used to need special handling:
  - **Qwen 2.5** (e.g., `Qwen/Qwen2.5-0.5B-Instruct` — ungated, small). BPE + ByteLevel. Widely deployed; Python transformers tests this family directly (`tests/models/qwen2/test_tokenization_qwen2.py`).
  - **Cohere / Aya** (ungated tokenizer-only community re-upload). BPE. `cohere_asr` is on Python v5's force-map list and has a dedicated upstream tokenization test — direct validation of the "class soup bypassed" claim.
  - **ModernBERT** (e.g., `answerdotai/ModernBERT-base` — ungated). WordPiece. On Python v5's force-map list (`modernbert`); notably, Python has *no* dedicated tokenization test for it because `TokenizersBackend` handles it class-agnostically — exactly the shape of this redesign.
  - **Gemma 2** (ungated community re-upload, e.g., `mlx-community/gemma-2-2b-it-4bit`). SentencePiece-BPE. Newer post-processor layout than `pcuenq/gemma-tokenizer` (Gemma 1-era, currently the only Gemma tested).
  - **Mistral v0.3+** (ungated community re-upload, e.g., `mlx-community/Mistral-7B-Instruct-v0.3-4bit`). BPE. Exercises the Mistral family's function-calling `added_tokens` structure, distinct from Llama 2.

### Out of scope

Upstream libraries have extensive coverage for features the Swift API does not expose:

- **Alignment methods** (`word_ids`, `token_to_word`, `word_to_tokens`, `token_to_chars`, `char_to_token`, `char_to_word`, `word_to_chars`, `token_to_sequence`, offsets mapping) — `PreTrainedTokenizer.encode` returns `[Int]`, not an `Encoding` with offsets.
- **Pair and batch encoding** — the Swift API is single-string.
- **Training, streaming decode, serialization round-trip** — not exposed.

Adding corresponding tests would require expanding the public API, which is outside the scope of this redesign.

## 7. Risks

- **The DeepSeek cliff.** DeepSeek repos that ship `post_processor.type = "ByteLevel"` in `tokenizer.json` together with `add_bos_token: true` in `tokenizer_config.json` and `tokenizer_class: "LlamaTokenizerFast"` will lose the leading bos under canonical behavior. Confirmed affected repos (as of investigation):
  - `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` / `-7B` / `-14B` / `-32B`
  - `deepseek-ai/DeepSeek-V3`
  - `deepseek-ai/DeepSeek-V2-Lite`
  - `deepseek-ai/deepseek-llm-7b-chat`
  - `deepseek-ai/deepseek-coder-6.7b-instruct`
  
  Canonical Python v5 and upstream Rust both also fail to add bos for these repos in practice. The Python mechanism is subtler than a direct force-map, and splits on whether the repo ships an `auto_map` entry in `tokenizer_config.json`:
  - **No `auto_map`:** the class-mismatch branch at `tokenization_auto.py:709-727` fires (its first precondition is `tokenizer_auto_map is None`). The force-map in §1.1 makes the mapped class name disagree with the repo's `tokenizer_class: "LlamaTokenizerFast"`, so the branch is entered and the specialized `LlamaTokenizer` is selected at line 724.
  - **With `auto_map`:** the mismatch branch is skipped. Lines 759-762 then disable remote code for model_types in the force-map set and clear `tokenizer_auto_map`; the later `tokenizer_class` fallback at `788-799` resolves the same `LlamaTokenizer`.
  
  Either way, `LlamaTokenizer` is selected, and it inherits the base `_should_update_post_processor` check. For a config with `post_processor.type = "ByteLevel"` and no user-passed bos/eos kwargs, both conditions are false and no bos injection runs. Outcome: no leading bos from Python v5 regardless of which dispatch branch the repo takes. We are not diverging from upstream — we're aligning with the same net behavior.
  
  Community re-uploads have already corrected their copies (verified: `mlx-community/DeepSeek-R1-Distill-Qwen-7B` and `unsloth/DeepSeek-R1-Distill-Qwen-7B` ship `post_processor.type = "TemplateProcessing"` with bos baked in). Users hitting the cliff have alternatives without requiring us to reintroduce a compat shim. The `deepSeekPostProcessor` test already describes its current expectation as a "post-processor override" — i.e., knowingly non-canonical. Under the redesign, the test expectation updates to the canonical value, matching what Python v5 and upstream Rust actually produce.
- **Removed public APIs.** `AutoTokenizer.register`, the `strict:` parameter, the `BertTokenizer` class name (plus `BasicTokenizer`), and two `TokenizerError` cases are breaking. Acceptable: the project is small, these are not documented in the README, and consistency with upstream is the goal.
- **Latent decode-fallback divergence exposed by WordLevel.** When `tokenizer.json` has no `decoder`, upstream Rust joins tokens with `" "` (`tokenizers/src/tokenizer/mod.rs:914-918`). Swift currently joins with `""` (`SwiftTokenizerBackend.swift:315`). The bug has been latent because every algorithm in use today (BPE / WordPiece / Unigram) ships a decoder in its `tokenizer.json`. WordLevel fixtures typically don't, so the redesign both exposes the bug and needs it fixed. Fix is part of §8.B; WordLevel fixtures deliberately omit the `decoder` field so the fallback path is exercised.
- **No expected performance impact.** Dispatch changes only algorithm selection. The deleted shims ran at load time.

## 8. Implementation checklist

Single PR, no staged rollout. The sections below decompose the change by concern for reviewability; they are not independently buildable checkpoints. Concretely: A rewrites `BertTokenizer` as `WordPieceTokenizer`, but the `knownTokenizers` registry in `SwiftTokenizerBackend.swift:44-66` references `BertTokenizer.self` until E deletes it; D's shim deletion touches factory code that E rewrites; G removes error cases that `TokenizerCompatibility.swift` still imports until J deletes that file. In practice A–J must land atomically in one commit. The ordering is a reading order for the diff, not a sequence of passing builds.

### Pre-refactor baseline

Write these against current `main` and land them separately (a standalone commit or PR) before starting A–L. They lock in current behavior so regressions in the refactor surface as test diffs rather than silently.

- [x] Add the Swift/Rust parity test against current behavior (§6). Baseline may show existing backend disagreements on shim-dependent models; that's fine — the post-refactor run must not add new ones, and the known-diff entries get removed once the shims are gone.
- [x] Extend `TokenizerTests.tokenizer` with a shared multi-script roundtrip sample and record expected outputs from current behavior (§6). After the refactor, expect zero deltas except on DeepSeek (§7 cliff) and NLLB (strict-mode removal).

### A. WordPiece rewrite

- [x] `git mv Sources/Tokenizers/BertTokenizer.swift Sources/Tokenizers/WordPieceTokenizer.swift`
- [x] Rewrite the file: replace `BertTokenizer` / `BasicTokenizer` / `WordpieceTokenizer` with a single `WordPieceTokenizer` conforming to `PreTrainedTokenizerModel`. Read `unk_token`, `continuing_subword_prefix`, and `max_input_chars_per_word` from `tokenizerData.model` (defaults `"[UNK]"` / `"##"` / `100`, matching upstream). No internal normalizer or pre-tokenizer — the backend pipeline handles those from `tokenizer.json`.
- [x] Delete `BasicTokenizer` (now unused).
- [x] Delete `Tests/TokenizersTests/BertTokenizerTests.swift` along with its SQuAD fixtures (`dev-v1.1.json`, `tokenized_questions.json`, `basic_tokenized_questions.json`, `question_tokens.json`, `bert-vocab.txt`). Coverage is retained via the existing `bertUncased` / `bertCased` hub-integration tests and new algorithm-dispatch tests in `AlgorithmTests.swift`; the SQuAD stress sample was redundant with those and its fixtures were tied to the deleted direct-construction API.
- [x] Update `Package.swift` `tokenizerSwiftBackendSources`.

### B. New algorithm + decode-fallback fix

- [x] Add `Sources/Tokenizers/WordLevelTokenizer.swift` (vocab lookup + unk fallback, conforming to `PreTrainedTokenizerModel`)
- [x] Add to `tokenizerSwiftBackendSources`
- [x] Fix `SwiftTokenizerBackend.decode` for the no-decoder case only. The join separator is selected based on whether a decoder ran: if `decoder == nil`, join the raw token strings with `" "` (matching upstream Rust at `tokenizers/src/tokenizer/mod.rs:914-918`); if a decoder ran (e.g., ByteLevel produced final text parts), keep the `""` join so ByteLevel-style output doesn't regress.
- [x] Add a WordLevel test that writes an inline tokenizer.json fixture (no `decoder` field, `clean_up_tokenization_spaces: false` in the sidecar) to a temp directory and exercises encode → decode via `AutoTokenizer.from(directory:)`. Lives in `Tests/TokenizersTests/AlgorithmTests.swift`. Generating the fixture at test time rather than committing it under `Resources/` avoids SPM's `.process(...)` resource-name collision with the existing top-level `tokenizer.json` fixture.

### C. Added-token matching parity

Swift currently handles only `content` / `id` / `special` / `lstrip` / `rstrip` for entries in `tokenizer.json`'s `added_tokens` array (`SwiftTokenizerBackend.swift:153-196` — accessed there via `Config`'s camelCase key `addedTokens`). Upstream Rust also honors `single_word` and `normalized` (`tokenizers/src/tokenizer/added_vocabulary.rs:21,27`). Without this, the "trust `tokenizer.json`" claim silently drops two fields. Required alongside the added-token tests in §6.

- [x] Extend the added-token parser to read `single_word` and `normalized` in addition to the fields handled today. Done via a new `AddedTokenInfo` struct inside `SwiftTokenizerBackend`; `parseAddedTokens` still returns the lightweight `(tokens, special)` split needed by factories.
- [x] Update the added-tokens matcher to honor `single_word`: tokens with `single_word: true` are discarded when an ASCII word character (`[_0-9A-Za-z]`, matching the `regex` crate's `\w`) sits immediately outside the *outer* match range (after any lstrip/rstrip whitespace consumption). Matching uses `NSRegularExpression.matches(in:)` + post-filter rather than `\b`-style anchors, avoiding the edge cases with Unicode combining marks and tokens that contain non-word characters themselves (e.g. `<mask>`).
- [x] Update the added-tokens matcher to honor `normalized`: non-normalized tokens match against raw input (phase 1), normalized tokens match against the normalized form of each remaining non-match section (phase 2). Upstream splits these into two matching passes; Swift does the same with per-pass regexes.
- [x] Coverage for the interaction with `lstrip` / `rstrip` / `single_word` / `normalized` lives in `AlgorithmTests.addedTokenBehaviorFlags` (Swift-only — Rust delegates the entire flow to upstream's `added_vocabulary`, which already has its own coverage).

### D. Delete shims

- [x] Swift: delete `LlamaSwiftTokenizerBackend`, `LlamaTokenizerConfig`, and `SwiftAutoTokenizerFactory.makeLlamaTokenizer` from `SwiftTokenizerBackend.swift`
- [x] Swift: delete the decorative `private final class T5Tokenizer: UnigramTokenizer {}` alias
- [x] Rust: delete `LlamaPostProcessorCompatibility`, `llama_post_processor_compatibility`, `resolve_optional_compatibility_token_id`, and `normalized_tokenizer_name` in `tokenizer_core.rs`
- [x] Rust: delete the on-top token-ID injection in `encode`; `encode` now simply returns what `self.tokenizer.encode(...)` produces
- [x] Rust: remove `llama_post_processor_compatibility` field from `TokenizerCore`

### E. Dispatch rewrite

- [x] Swift `SwiftAutoTokenizerFactory.from`: dispatch on `tokenizerData["model"]["type"].string()` (`BPE` / `WordPiece` / `Unigram` / `WordLevel`), throwing `TokenizerError.unsupportedModelType(...)` otherwise. Fold the former `makeLlamaTokenizer` branch into the regular path.
- [x] Swift: delete `RegisteredTokenizerStore`, `TokenizerModel.registerTokenizer`, `TokenizerModel.tokenizerClass(for:)`, and the `knownTokenizers` dictionary
- [x] Swift: delete the `public extension AutoTokenizer { static func register... }` block
- [x] Swift: update `TokenizerModel.from` to the four-branch dispatch. Deviation from §3.1: a shape-based legacy-untagged fallback (`inferLegacyModelType`) was added because `bert-base-uncased` and other older BERT-family repos ship `tokenizer.json` without a `model.type` tag. The fallback mirrors upstream's `ModelWrapper::Deserialize` at `tokenizers/src/models/mod.rs:125-133` (BPE first, then WordPiece, WordLevel, Unigram) via shape hints (presence of `merges`, `continuing_subword_prefix`, `max_input_chars_per_word`, and vocab array-vs-dict). Strict rejection still fires when the shape matches none.

### F. Strict mode / allow-list removal

- [x] Drop `strict: Bool = true` from `AutoTokenizer.from(directory:)` in the facade
- [x] Swift: remove `strict` threading from `SwiftAutoTokenizerDirectoryLoader.load` / `loadTokenizerCore` / `SwiftAutoTokenizerFactory.from`
- [x] Rust side: remove `strict` threading from `RustAutoTokenizerDirectoryLoader.load` / `loadTokenizerCore`; delete `validateStrictCompatibility`

### G. Error enum

- [x] Add `TokenizerError.unsupportedModelType(String)` with a useful `errorDescription`
- [x] Remove `TokenizerError.unsupportedTokenizer(String)` and `.missingTokenizerClassInConfig`
- [x] Audit Rust FFI error codes in `RustFFI.tokenizerError` for any dead references. Code 7 (`missingChatTemplate`) had no producer on the Rust side (the Swift facade throws it before any FFI call) and was removed.

### H. `RuntimeConfiguration` cleanup

- [x] Swift: remove `tokenizerClass`, `modelType`, `addBosToken`, `addEosToken`, `legacy` from `TokenizerRuntimeConfiguration`; update initializer and `Codable` conformance
- [x] Rust: remove the same fields from `RuntimeConfiguration` in `sidecars.rs`; update serde derives
- [x] Swift: remove readers that fed these fields (`tokenizerConfig.addBosToken.boolean(or: false)` etc.). Since all consumers are being deleted in D, most readers are gone anyway; this step is a final sweep.

### I. Sidecar loader cleanup

- [x] Swift `AutoTokenizerDirectorySidecars.load`: remove the `config.json` merge that backfilled `tokenizer_class`
- [x] Rust `sidecars.rs`: remove the `config.json` read fallbacks for `tokenizer_class` / `model_type`; if `config.json` is no longer read for anything, remove the code path
- [x] Keep chat-template override loading (`chat_template.jinja` / `chat_template.json`) unchanged

### J. Delete `TokenizerCompatibility.swift`

- [x] Delete the file entirely (`git rm`)
- [x] Remove it from `tokenizerCoreSources` in `Package.swift`

### K. Tests

Required:

- [x] Update `TokenizerTests.deepSeekPostProcessor` expected IDs and comment (see §6)
- [x] Rewrite `TokenizerTests.nllbTokenizer` to just load and encode
- [x] Delete `TokenizerTests.concurrentTokenizerRegistration`
- [x] Add the WordLevel test (§6)
- [x] Add the `unsupportedModelType` error test (§6). Swift-only because Rust's upstream crate throws a different error path for malformed `tokenizer.json`; the Swift-specific dispatch guard is what the test is covering.
- [x] Add the algorithm dispatch test — one inline fixture per `model.type` (§6). Swift-only because dispatch happens on the Swift side only; Rust delegates the whole flow to the upstream crate.
- [x] Add added-tokens behavior tests — lstrip, rstrip, single_word, normalized (§6). Swift-only — exercises `SwiftTokenizerBackend`'s matcher; Rust's delegation is covered by upstream.
- [x] Re-record the multi-script roundtrip expectations for DeepSeek and NLLB (baseline captured in Pre-refactor); confirm every other model's baseline still holds. Swift/Rust Llama-2 multi-script values stayed byte-identical post-refactor, so the Swift-side entry didn't need re-recording.
- [x] Re-run the Swift/Rust parity test (baseline captured in Pre-refactor); remove any known-diff entries that existed only because of the deleted shims. Swift and Rust DeepSeek parity values converge on the canonical `[15191, 525, 498, 30]` once the rebuilt xcframework is in use.

Recommended:

- [x] Add TemplateProcessing functional tests — single/pair expansion, special-token substitution (§6). Lives in `PostProcessorTests.TemplateProcessingTests`.
- [x] Add Unigram algorithm unit tests (§6). Lives in `AlgorithmTests` alongside the WordLevel fixture.
- [x] Add compact hub-integration tests for Qwen 2.5, Cohere/Aya, ModernBERT, Gemma 2, and Mistral v0.3+ (§6). For Cohere/Aya, settled on `mlx-community/aya-expanse-8b-4bit` (an ungated community re-upload) after the `CohereLabs/aya-expanse-8b` repo returned no commit hash for `config.json`.

### L. Ancillaries

- [x] Remove `strict: true` from `Tests/Benchmarks/SubsystemBenchmarkSupport.swift`
- [x] Scan README for any stale references; update any that exist (README was already clean — no references to `strict:`, `AutoTokenizer.register`, or `BertTokenizer`).
- [x] Run `cargo fmt -- --check` and `cargo clippy` on the Rust crate. Applied `cargo fmt` across the crate (pre-existing formatting drift unrelated to the refactor) and fixed one pre-existing `collapsible_if` clippy warning in `apply_chat_template`.
- [x] Run the Swift test suite under both the Swift and Rust traits; confirm parity. Swift trait 130/130, Rust trait 45/45 (with `TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH` pointed at a locally-built xcframework — the published `tokenizers-rust-0.3.1` artifact still ships the old FFI schema and shim).

## Outstanding after this branch

- **Rebuild and release the `TokenizersRust` xcframework** from the updated Rust source in this branch (the breaking FFI schema change means bumping at least the minor tag — likely `tokenizers-rust-0.4.0`). Update `Package.swift`'s binary-target URL + checksum to point at the new release. Until that lands, anyone running `swift test --traits Rust` without the `TOKENIZERS_RUST_LOCAL_XCFRAMEWORK_PATH` override will fail to deserialize the trimmed `RuntimeConfiguration` JSON.
- **Optional: CHANGELOG / README migration section.** §10 of this doc already describes the breaking public-API changes (`AutoTokenizer.register` removed, `strict:` parameter removed, `BertTokenizer` / `BasicTokenizer` removed, two `TokenizerError` cases renamed, DeepSeek-family loses leading bos). Worth turning into a short user-facing note.

## 9. Non-goals

- Mirroring Python transformers' per-class `__init__` rebuilds (~45 subclasses of `TokenizersBackend` that construct `normalizer` / `pre_tokenizer` / `decoder` from code rather than from `tokenizer.json`). For well-formed configs, what the class constructs matches what `tokenizer.json` contains; the divergence only bites malformed configs, which is out of scope.
- Per-class `update_post_processor` overrides (Splinter, MLuke, Cohere-via-`_post_init` for malformed configs). Same rationale — out of scope.
- `convert_to_native_format` truncation / padding passthrough from `tokenizer.json` (`tokenization_utils_tokenizers.py:394-411`). Not supported today; not in scope.
- `fast_tokenizer_files` alternate-file resolution in `tokenizer_config.json`. Not supported today; not in scope.
- Redesigning component factories (`NormalizerFactory`, `PreTokenizerFactory`, `PostProcessorFactory`, `DecoderFactory`). Already algorithm/type-keyed on `tokenizer.json` fields.
- Changing the backend-selection mechanism (Swift trait vs Rust trait).
- Touching chat-template rendering or the Jinja pipeline.
- Unifying `performsCleanup` between backends (flagged in §5.4).

## 10. Migration for users

These are breaking changes. A major version bump is appropriate (pre-1.0: `0.X` → `0.Y`). The breakage is bounded and covered in full below so that a CHANGELOG entry and a short README migration section can redirect users.

### 10.1 Source-level changes

| Before | After |
|--------|-------|
| `AutoTokenizer.from(directory: url, strict: true)` | `AutoTokenizer.from(directory: url)` — drop the `strict:` argument |
| `AutoTokenizer.from(directory: url, strict: false)` | `AutoTokenizer.from(directory: url)` — the non-strict path was only reachable via `strict: false`; the default behavior is now always "load what `tokenizer.json` says" |
| `AutoTokenizer.register(MyTokenizer.self, for: "MyName")` | Removed. Algorithm dispatch is driven by `tokenizer.json`'s `model.type`; if you need a new algorithm, it belongs in the package, not as a runtime registration. |
| `BertTokenizer(vocab:merges:tokenizeChineseChars:...)` | No direct equivalent. `BertTokenizer` is replaced by `WordPieceTokenizer`, which is a pure WordPiece model step with a config-driven initializer (`tokenizerConfig` / `tokenizerData` / `addedTokens` / `vocab` / `merges`), not a self-contained BERT pipeline. Load via `AutoTokenizer.from(directory:)` against a directory containing `tokenizer.json`; the backend assembles normalizer, pre-tokenizer, model, and decoder from that file. Direct construction from a bare vocab is not supported. |
| `catch TokenizerError.unsupportedTokenizer(let name) { ... }` | `catch TokenizerError.unsupportedModelType(let type) { ... }` — raised when `tokenizer.json`'s `model.type` is missing or unrecognized. |
| `catch TokenizerError.missingTokenizerClassInConfig { ... }` | Never thrown — the field is no longer consulted. If `tokenizer.json` is missing entirely you get `.missingConfig` as before. |

### 10.2 Behavioral changes

**DeepSeek-family repos lose the leading bos token.** The following repos ship a `tokenizer.json` whose `post_processor.type = "ByteLevel"` and rely on the pre-v5 Llama compatibility shim to inject bos. Post-redesign, the output no longer contains the leading bos:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- `deepseek-ai/DeepSeek-V3`
- `deepseek-ai/DeepSeek-V2-Lite`
- `deepseek-ai/deepseek-llm-7b-chat`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

This matches canonical Python transformers v5 and upstream `huggingface/tokenizers` in practice. Python v5's dispatch for these repos depends on whether `tokenizer_config.json` ships a `tokenizer_class` value (and which one) — see §7's detailed trace — but the net outcome converges: no leading bos, because the generic `_should_update_post_processor` check in `TokenizersBackend` is inherited and isn't triggered for a config whose `tokenizer.json` already has a (non-bos-adding) post-processor.

Three options for affected users:

1. **Use a community re-upload with a correct `tokenizer.json`.** Verified corrected: `mlx-community/DeepSeek-R1-Distill-Qwen-7B`, `unsloth/DeepSeek-R1-Distill-Qwen-7B`. Other quantization communities have generally done the same for their copies.
2. **Fix the `tokenizer.json` locally.** Replace the top-level `post_processor: { type: "ByteLevel", ... }` with a `Sequence` wrapping `[ByteLevel, TemplateProcessing]`, where `TemplateProcessing` prepends the bos token. This is the structure modern Llama 3.x repos already ship.
3. **Prepend the bos token at the application layer** after calling `encode`.

**Unknown `tokenizer_class` no longer silently falls back to BPE.** The previous Swift backend dispatched on `tokenizer_class` and fell back to `BPETokenizer` for unknown names (correct for most Qwen/Cohere/Gemma variants, silently wrong for WordPiece and Unigram models). Post-redesign, dispatch is driven by `tokenizer.json`'s `model.type` — the authoritative field. If `model.type` is present and valid, tokenization is correct regardless of `tokenizer_class`. If it's missing or unrecognized, `TokenizerError.unsupportedModelType` is thrown.

**Strict-mode throws become successful loads.** Models whose `tokenizer_class` was not on the `rustSupportedTokenizerNames` allow-list (NLLB, and others that aren't on the list) used to throw `TokenizerError.unsupportedTokenizer` under the default `strict: true`. They now load successfully. Code that catches that exception and handles the failure will no longer reach the `catch` block. Most callers get a strict upgrade (fewer exceptions to handle).

### 10.3 Not breaking

Any model whose `tokenizer.json` has a correct `post_processor` continues to produce the same output. The overwhelming majority of models in the ecosystem are well-formed in this sense: Llama 2 / 3.x, Mistral, Mixtral, Gemma 2 / 3, Qwen 2 / 2.5 / 3, Phi-3.x, Phi-4, Whisper, BERT, RoBERTa, XLM-RoBERTa, NLLB, T5, and all models with `TemplateProcessing` or a `Sequence` that includes a bos-prepending processor.

Unchanged API: `Tokenizer`, `PreTrainedTokenizer`, `TokenizingModel`, `PreTrainedTokenizerModel`, `TokenizerVocab`, `TokenizerMerges`, `ChatTemplateArgument`, all chat-template functionality.

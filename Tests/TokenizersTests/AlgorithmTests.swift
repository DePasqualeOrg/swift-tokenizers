// Copyright © Anthony DePasquale

import Foundation
import Testing

@testable import Tokenizers

#if TOKENIZERS_SWIFT_BACKEND
@testable import TokenizersSwiftBackend
#endif

/// Tests for algorithm-level behavior exercised through tiny on-disk fixtures. These
/// cover cases the hub-integration suite can't reach on its own: the WordLevel
/// decode-fallback path, `TokenizerError.unsupportedModelType`, per-algorithm
/// dispatch on `tokenizer.json`'s `model.type`, and added-token behavior flags.
@Suite("Algorithm", .serialized)
struct AlgorithmTests {
    // MARK: - Fixture helpers

    /// Writes `tokenizer.json` (and optionally `tokenizer_config.json`) under a fresh
    /// temporary directory, returns the directory. Tiny local fixtures live here as
    /// inline JSON rather than resource files so they don't collide with the existing
    /// top-level `tokenizer.json` / `tokenizer_config.json` offline fixture (SPM's
    /// `.process(...)` rule flattens resource names).
    private static func writeFixtureDirectory(
        tokenizerJSON: String,
        tokenizerConfigJSON: String? = nil
    ) throws -> URL {
        let base = FileManager.default.temporaryDirectory
            .appending(component: "swift-tokenizers-algorithm-tests")
            .appending(component: UUID().uuidString)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try tokenizerJSON.data(using: .utf8)!.write(
            to: base.appending(component: "tokenizer.json")
        )
        if let tokenizerConfigJSON {
            try tokenizerConfigJSON.data(using: .utf8)!.write(
                to: base.appending(component: "tokenizer_config.json")
            )
        }
        return base
    }

    // MARK: - WordLevel end-to-end

    private static let wordLevelTokenizerJSON: String = """
        {
          "version": "1.0",
          "added_tokens": [
            { "id": 0, "content": "[UNK]", "special": true }
          ],
          "normalizer": { "type": "Lowercase" },
          "pre_tokenizer": { "type": "Whitespace" },
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "hello": 1,
              "world": 2,
              "the": 3,
              "quick": 4,
              "brown": 5,
              "fox": 6,
              "jumps": 7,
              "over": 8,
              "lazy": 9,
              "dog": 10
            },
            "unk_token": "[UNK]"
          }
        }
        """

    private static let wordLevelTokenizerConfigJSON: String = """
        {
          "unk_token": "[UNK]",
          "clean_up_tokenization_spaces": false
        }
        """

    @Test
    func wordLevelEncodeDecode() async throws {
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: Self.wordLevelTokenizerJSON,
            tokenizerConfigJSON: Self.wordLevelTokenizerConfigJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        let expected = [3, 4, 5, 6, 7, 8, 3, 9, 10]
        let encoded = tokenizer.encode(text: "The quick brown fox jumps over the lazy dog")
        #expect(encoded == expected)

        // No decoder in the fixture → upstream falls back to joining with " ". The
        // fixture also sets `clean_up_tokenization_spaces: false` so the space-join
        // survives the backend's cleanup pass.
        let decoded = tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false)
        #expect(decoded == "the quick brown fox jumps over the lazy dog")

        // Out-of-vocabulary words fall through to the unknown token.
        #expect(tokenizer.encode(text: "Hello unknown world") == [1, 0, 2])
    }

    // MARK: - Unsupported model.type

    #if TOKENIZERS_SWIFT_BACKEND
    @Test
    func unsupportedModelTypeThrows() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": { "type": "Nonsense", "vocab": {} }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)

        do {
            _ = try await AutoTokenizer.from(directory: directory)
            Issue.record("Expected AutoTokenizer.from to throw")
        } catch let TokenizerError.unsupportedModelType(type) {
            #expect(type == "Nonsense")
        } catch {
            Issue.record("Wrong error: \(error)")
        }
    }

    @Test
    func missingModelTypeThrows() async throws {
        // A model object with no `type` and none of the shape hints that the legacy
        // untagged fallback uses (no `vocab`, no `merges`, no WordPiece-specific
        // fields) should throw with an empty-string model type.
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": { "unused_field": true }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)

        do {
            _ = try await AutoTokenizer.from(directory: directory)
            Issue.record("Expected AutoTokenizer.from to throw")
        } catch let TokenizerError.unsupportedModelType(type) {
            #expect(type.isEmpty)
        } catch {
            Issue.record("Wrong error: \(error)")
        }
    }
    #endif

    // MARK: - Algorithm dispatch (Swift backend)

    #if TOKENIZERS_SWIFT_BACKEND
    @Test
    func dispatchesOnModelType() async throws {
        // BPE
        let bpeJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "h": 0, "i": 1 },
                "merges": []
              }
            }
            """
        let bpeDir = try Self.writeFixtureDirectory(tokenizerJSON: bpeJSON)
        let bpe = try await AutoTokenizer.from(directory: bpeDir)
        #expect((bpe as? PreTrainedTokenizer)?.model is BPETokenizer)

        // WordPiece
        let wpJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "vocab": { "[UNK]": 0, "hi": 1 },
                "unk_token": "[UNK]"
              }
            }
            """
        let wpDir = try Self.writeFixtureDirectory(tokenizerJSON: wpJSON)
        let wp = try await AutoTokenizer.from(directory: wpDir)
        #expect((wp as? PreTrainedTokenizer)?.model is WordPieceTokenizer)

        // Unigram
        let uniJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [["<unk>", 0.0], ["hi", -1.0]]
              }
            }
            """
        let uniDir = try Self.writeFixtureDirectory(tokenizerJSON: uniJSON)
        let uni = try await AutoTokenizer.from(directory: uniDir)
        #expect((uni as? PreTrainedTokenizer)?.model is UnigramTokenizer)

        // WordLevel
        let wlDir = try Self.writeFixtureDirectory(
            tokenizerJSON: Self.wordLevelTokenizerJSON,
            tokenizerConfigJSON: Self.wordLevelTokenizerConfigJSON
        )
        let wl = try await AutoTokenizer.from(directory: wlDir)
        #expect((wl as? PreTrainedTokenizer)?.model is WordLevelTokenizer)
    }
    #endif

    // MARK: - Unigram algorithm

    /// Build a minimal Unigram fixture. `vocab` is `(token, score)` pairs; index 0 is
    /// the unknown token by convention. Higher scores are preferred during
    /// segmentation (the algorithm maximizes summed log-probability).
    private static func unigramFixture(vocab: [(String, Double)]) throws -> URL {
        let vocabEntries = vocab.map { "[\"\($0.0)\", \($0.1)]" }.joined(separator: ",")
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "Unigram",
                "unk_id": 0,
                "vocab": [\(vocabEntries)]
              }
            }
            """
        let configJSON: String = """
            { "unk_token": "<unk>", "clean_up_tokenization_spaces": false }
            """
        return try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
    }

    @Test
    func unigramPrefersLongestHighScoreSpan() async throws {
        // `ab` (score −1.0) beats `a` + `b` (−3.0 − 3.0 = −6.0).
        let directory = try Self.unigramFixture(vocab: [
            ("<unk>", -20.0),
            ("a", -3.0),
            ("b", -3.0),
            ("ab", -1.0),
        ])
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["ab"])
    }

    @Test
    func unigramFallsBackToCharacterSplit() async throws {
        // With the merged token's score far below the individual characters,
        // segmentation chooses the per-character split.
        let directory = try Self.unigramFixture(vocab: [
            ("<unk>", -20.0),
            ("a", -0.5),
            ("b", -0.5),
            ("ab", -5.0),
        ])
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["a", "b"])
    }

    @Test
    func unigramSegmentsAcrossMultiplePieces() async throws {
        // Standard SentencePiece-style segmentation: multiple overlapping pieces, the
        // algorithm picks the highest-scoring decomposition.
        let directory = try Self.unigramFixture(vocab: [
            ("<unk>", -20.0),
            ("he", -1.0),
            ("ll", -1.0),
            ("o", -1.0),
            ("hell", -5.0),
            ("lo", -5.0),
        ])
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        // "hello" can split as he+ll+o (−3) or hell+o (−6) or he+l+l+o (fails — no "l")
        // or hell+o (−6). The algorithm picks he+ll+o since −3 > −6.
        #expect(tokenizer.tokenize(text: "hello") == ["he", "ll", "o"])
    }

    @Test
    func unigramUnknownCharactersGoToUnk() async throws {
        let directory = try Self.unigramFixture(vocab: [
            ("<unk>", -20.0),
            ("h", -1.0),
            ("i", -1.0),
        ])
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        // Upstream Unigram fuses consecutive unknowns by concatenating their raw
        // text spans into a single token (see `Unigram::tokenize` in
        // `tokenizers/src/models/unigram/model.rs`). Both backends produce `"xy"`
        // as one fused unknown between the known `h` and `i`.
        #expect(tokenizer.tokenize(text: "hxyi") == ["h", "xy", "i"])
        // Any unknown character still maps to the unk id (0).
        #expect(tokenizer.convertTokenToId("x") == 0)
        #expect(tokenizer.convertTokenToId("y") == 0)
    }

    // MARK: - Added-token behavior

    /// Exercises the `lstrip`, `rstrip`, `single_word`, and `normalized` flags on
    /// entries in `tokenizer.json`'s `added_tokens` array, mirroring the relevant
    /// cases from upstream `tokenizers/tests/added_tokens.rs`. Swift-only because
    /// this exercises `SwiftTokenizerBackend`'s matcher; Rust delegates the entire
    /// flow to the upstream crate, which already has its own coverage.
    #if TOKENIZERS_SWIFT_BACKEND
    @Test
    func addedTokenBehaviorFlags() async throws {
        // Fixture: a simple WordLevel model with four added tokens exercising each
        // behavior flag.
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": true, "lstrip": true, "normalized": false },
                { "id": 2, "content": "<pad>", "special": true, "rstrip": true, "normalized": false },
                { "id": 3, "content": "<word>", "special": true, "single_word": true, "normalized": false },
                { "id": 4, "content": "<lc>", "special": false, "normalized": true }
              ],
              "normalizer": { "type": "Lowercase" },
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": {
                  "[UNK]": 0,
                  "<mask>": 1,
                  "<pad>": 2,
                  "<word>": 3,
                  "<lc>": 4,
                  "hello": 5,
                  "world": 6,
                  "abc": 7,
                  "def": 8
                },
                "unk_token": "[UNK]"
              }
            }
            """
        let configJSON: String = """
            { "unk_token": "[UNK]", "clean_up_tokenization_spaces": false }
            """
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        // lstrip: `<mask>` eats the whitespace on its left during matching. Upstream
        // stores the widened slice (leading whitespace + the added token) as the
        // Token's value, so the emitted token carries the consumed whitespace even
        // though it encodes to the `<mask>` added-token id.
        #expect(tokenizer.tokenize(text: "hello    <mask> world") == ["hello", "    <mask>", "world"])

        // rstrip: `<pad>` eats the whitespace on its right; the widened slice keeps it.
        #expect(tokenizer.tokenize(text: "hello <pad>    world") == ["hello", "<pad>    ", "world"])

        // single_word: `<word>` must not be adjacent to word characters. The first
        // call matches because spaces bracket the token; the second does not, so the
        // substring is folded into the unknown token.
        #expect(tokenizer.tokenize(text: "abc <word> def") == ["abc", "<word>", "def"])
        #expect(tokenizer.tokenize(text: "abc<word>def") == ["[UNK]"])

        // normalized: `<lc>` is registered with normalized=true. The normalizer here
        // lowercases its input, so upper-case `<LC>` survives phase 1 (non-normalized
        // pass finds nothing), gets lowercased to `<lc>`, then phase 2 matches it.
        #expect(tokenizer.tokenize(text: "hello <LC> world") == ["hello", "<lc>", "world"])
    }

    // MARK: - Upstream-parity regression tests
    //
    // These mirror specific upstream tests from `huggingface/tokenizers` to anchor
    // our added-tokens matcher and WordPiece decoder against the canonical Rust
    // implementation. They use WordLevel / WordPiece fixtures rather than upstream's
    // ByteLevel BPE setup (which requires the gpt2 vocab), but assert the same
    // semantic behavior.

    // MARK: - tokenize() runs post-processor

    /// `tokenize()` must run the post-processor with `addSpecialTokens=false`,
    /// matching Python transformers v5 (`TokenizersBackend.tokenize` in
    /// `src/transformers/tokenization_utils_tokenizers.py`) and the Rust backend.
    /// For a model with `TemplateProcessing` adding `[CLS]`/`[SEP]`, `tokenize()`
    /// suppresses them while `encode(addSpecialTokens: true)` includes them.
    @Test
    func tokenizeRunsPostProcessorWithSpecialsSuppressed() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "[CLS]", "special": true, "normalized": false },
                { "id": 2, "content": "[SEP]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                  { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
                  { "Sequence": { "id": "A", "type_id": 0 } },
                  { "SpecialToken": { "id": "[SEP]", "type_id": 0 } }
                ],
                "pair": [
                  { "SpecialToken": { "id": "[CLS]", "type_id": 0 } },
                  { "Sequence": { "id": "A", "type_id": 0 } },
                  { "SpecialToken": { "id": "[SEP]", "type_id": 0 } }
                ]
              },
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hi": 3, "there": 4 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "hi there") == ["hi", "there"])
        #expect(tokenizer.encode(text: "hi there", addSpecialTokens: true) == [1, 3, 4, 2])
        #expect(tokenizer.encode(text: "hi there", addSpecialTokens: false) == [3, 4])
    }

    // MARK: - Legacy untagged dispatch

    /// Untagged BPE (no `model.type`, `merges` present) must route to `BPETokenizer`.
    /// Upstream's `ModelWrapper` custom `Deserialize` impl in `tokenizers/src/models/mod.rs`
    /// tries BPE first in the untagged fallback; our shape-based hint must match.
    @Test
    func untaggedBPERoutedAsBPE() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "vocab": { "h": 0, "i": 1 },
                "merges": []
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect((tokenizer as? PreTrainedTokenizer)?.model is BPETokenizer)
    }

    /// Untagged WordPiece (no `model.type`, `continuing_subword_prefix` present)
    /// must route to `WordPieceTokenizer`. `bert-base-uncased`'s `tokenizer.json` is
    /// the canonical example — it ships without `model.type`.
    @Test
    func untaggedWordPieceRoutedAsWordPiece() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "vocab": { "[UNK]": 0, "hi": 1 },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect((tokenizer as? PreTrainedTokenizer)?.model is WordPieceTokenizer)
    }

    /// Untagged WordLevel (no `model.type`, dict vocab, no merges and no
    /// WordPiece-specific fields) must route to `WordLevelTokenizer`.
    @Test
    func untaggedWordLevelRoutedAsWordLevel() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "vocab": { "<unk>": 0, "hi": 1 },
                "unk_token": "<unk>"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect((tokenizer as? PreTrainedTokenizer)?.model is WordLevelTokenizer)
    }

    // MARK: - Unresolvable unk token

    /// WordLevel must throw at init when the configured unk token isn't in the
    /// vocab. Upstream `WordLevel::tokenize` (in `tokenizers/src/models/wordlevel/mod.rs`)
    /// raises `Error::MissingUnkToken` at tokenize time; our non-throwing tokenize
    /// signature can't bubble, so we check at init and turn a silent runtime crash
    /// into a clear init-time error.
    @Test
    func wordLevelThrowsWhenUnkMissing() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "a": 0, "b": 1 },
                "unk_token": "<unk>"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        await #expect(throws: TokenizerError.missingUnknownToken(model: "WordLevel")) {
            _ = try await AutoTokenizer.from(directory: directory)
        }
    }

    @Test
    func wordLevelSucceedsWhenUnkInVocab() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "<unk>": 0, "a": 1 },
                "unk_token": "<unk>"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "zzz") == ["<unk>"])
    }

    /// WordPiece has the same requirement (`WordPiece::tokenize` in
    /// `tokenizers/src/models/wordpiece/mod.rs`).
    @Test
    func wordPieceThrowsWhenUnkMissing() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "vocab": { "a": 0, "b": 1 },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        await #expect(throws: TokenizerError.missingUnknownToken(model: "WordPiece")) {
            _ = try await AutoTokenizer.from(directory: directory)
        }
    }

    /// Unigram must throw when `unk_id` is out of range for the vocab array
    /// (upstream `UnigramError::MissingUnkId` at `unigram/model.rs` is the equivalent).
    @Test
    func unigramThrowsWhenUnkIdOutOfRange() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "Unigram",
                "unk_id": 99,
                "vocab": [["<unk>", -10.0], ["a", -1.0]]
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        await #expect(throws: TokenizerError.missingUnknownToken(model: "Unigram")) {
            _ = try await AutoTokenizer.from(directory: directory)
        }
    }

    // MARK: - fuse_unk scoping

    /// `fuse_unk` is a Unigram-only concept upstream (a field on `Unigram` in
    /// `tokenizers/src/models/unigram/model.rs`; BPE is the only other model that
    /// serializes it, via `bpe/serialization.rs`). WordPiece and WordLevel must
    /// ignore the flag even when it's set in `tokenizer_config.json` — otherwise
    /// consecutive unks would collapse incorrectly.
    @Test
    func wordPieceIgnoresFuseUnkFromTokenizerConfig() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "vocab": { "[UNK]": 0, "a": 1 },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100
              }
            }
            """
        let configJSON = #"{ "fuse_unk": true }"#
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "xxx yyy zzz") == ["[UNK]", "[UNK]", "[UNK]"])
    }

    /// Upstream BPE (`tokenizers/src/models/bpe/serialization.rs`) serializes
    /// `fuse_unk` as a field of `tokenizer.json`'s `model` block. Setting the key
    /// at the `tokenizer_config.json` top level alone has no effect.
    @Test
    func bpeIgnoresFuseUnkFromTokenizerConfig() async throws {
        // Minimal BPE: no merges, empty vocab apart from `<unk>`. `tokenize("ab")`
        // falls through to byte-fallback (`<0x61>`, `<0x62>`), neither of which is
        // in vocab, so both resolve to unknown. With fusion off, two tokens come
        // out; with fusion on, they'd be concatenated into one.
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0 },
                "merges": [],
                "unk_token": "<unk>"
              }
            }
            """
        // `fuse_unk` set only on the wrong file must not trigger fusion.
        let configJSON = #"{ "fuse_unk": true, "unk_token": "<unk>" }"#
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        let tokens = tokenizer.tokenize(text: "ab")
        #expect(tokens.count == 2, "fuse_unk from tokenizer_config.json must not fuse consecutive unknowns")
    }

    /// The matching positive case: with `fuse_unk: true` in `tokenizer.json`'s
    /// `model` block, BPE fuses consecutive unknowns into a single token. Upstream
    /// `bpe/model.rs` extends a single `(unk_id, unk_len)` entry and maps the id
    /// back through `vocab_r`, so a fused run collapses to one copy of the unk
    /// token string (not N concatenated copies).
    @Test
    func bpeReadsFuseUnkFromTokenizerJsonModel() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0 },
                "merges": [],
                "unk_token": "<unk>",
                "fuse_unk": true
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["<unk>"])
    }

    /// Upstream BPE applies `fuse_unk` inside `BPE::tokenize`, so the fusion
    /// window is a single pre-tokenized word. Two words split by the
    /// Whitespace pre-tokenizer each OOV must stay separate, even when
    /// `fuse_unk: true` is set.
    @Test
    func bpeFuseUnkDoesNotCrossPreTokens() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0 },
                "merges": [],
                "unk_token": "<unk>",
                "fuse_unk": true
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "a b") == ["<unk>", "<unk>"])
    }

    /// Upstream BPE's OOV fallback reads `tokenizer.json.model.unk_token` only
    /// (`bpe/model.rs`). Python fast tokenizers match: the Rust backend's BPE
    /// model field is what drives emission, and `tokenizer_config.json.unk_token`
    /// is purely tokenizer-level metadata. When the two files disagree, Swift
    /// must still emit the model's unk, not the config's.
    @Test
    func bpeEmitsModelUnkEvenWhenTokenizerConfigDiffers() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<model-unk>", "special": true, "normalized": false },
                { "id": 1, "content": "<config-unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<model-unk>": 0, "<config-unk>": 1 },
                "merges": [],
                "unk_token": "<model-unk>"
              }
            }
            """
        let configJSON = #"{ "unk_token": "<config-unk>" }"#
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "x") == ["<model-unk>"])
        #expect(tokenizer.unknownToken == "<config-unk>")
    }

    /// Upstream `AddedVocabulary::split_with_indices` stores the widened match
    /// slice (lstrip/rstrip-consumed text) as the Token's value while keeping
    /// the added-token ID on the same Token. Python `transformers` fast and
    /// the Rust backend expose that value directly: `tokenize` returns the
    /// widened string, `encode` returns the added-token ID. Swift must keep
    /// both sides of that association to match.
    @Test
    func addedTokenLstripEmitsWidenedValueWithOverrideId() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": true, "lstrip": true, "rstrip": false, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "hi": 2, "there": 3 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        // The widened slice `" <mask>"` is not a vocab key; without the override
        // the ID would fall back to `[UNK]`. The override keeps encoding aligned
        // with upstream, which maps the widened string to the added-token ID.
        #expect(tokenizer.tokenize(text: "hi <mask> there") == ["hi", " <mask>", "there"])
        #expect(tokenizer.encode(text: "hi <mask> there", addSpecialTokens: false) == [2, 1, 3])
    }

    /// Positional ID-carrying across post-processing: two added tokens can
    /// share a value string (one phase-1 literal, one phase-2 normalized form)
    /// and still need to keep distinct IDs per position. Upstream Rust Tokens
    /// each carry their own ID, so `encode("day DAY")` returns `[2, 1]` — the
    /// phase-1 match wins first, then phase 2 matches the normalized `DAY`.
    /// A string-global override table would collapse both to whichever id
    /// happened to be inserted first, which is the exact regression codex
    /// flagged.
    @Test
    func addedTokenIdSurvivesPositionallyAcrossPhases() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "DAY", "special": false, "normalized": true },
                { "id": 2, "content": "day", "special": false, "normalized": false }
              ],
              "normalizer": { "type": "Lowercase" },
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "DAY": 1, "day": 2 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        #expect(tokenizer.tokenize(text: "day DAY") == ["day", "day"])
        #expect(tokenizer.encode(text: "day DAY", addSpecialTokens: false) == [2, 1])
        #expect(tokenizer.tokenize(text: "DAY day") == ["day", "day"])
        #expect(tokenizer.encode(text: "DAY day", addSpecialTokens: false) == [1, 2])
    }

    /// WordLevel's model unk (`tokenizer.json.model.unk_token`, default `"<unk>"`)
    /// must be present in vocab. A `tokenizer_config.json.unk_token` pointing at a
    /// token not in vocab is fine — it's user-facing metadata only, layered on
    /// top of the model's field by `PreTrainedTokenizer.unknownToken`.
    @Test
    func wordLevelLoadsWhenTokenizerConfigUnkIsNotInVocab() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "hello": 1 },
                "unk_token": "[UNK]"
              }
            }
            """
        let configJSON = #"{ "unk_token": "<|reserved|>" }"#
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "hello zzz") == ["hello", "[UNK]"])
        #expect(tokenizer.unknownToken == "<|reserved|>")
    }

    /// Same `fuse_unk` isolation rule for WordLevel.
    @Test
    func wordLevelIgnoresFuseUnkFromTokenizerConfig() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "a": 1 },
                "unk_token": "[UNK]"
              }
            }
            """
        let configJSON = #"{ "fuse_unk": true }"#
        let directory = try Self.writeFixtureDirectory(
            tokenizerJSON: tokenizerJSON,
            tokenizerConfigJSON: configJSON
        )
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "xxx yyy zzz") == ["[UNK]", "[UNK]", "[UNK]"])
    }

    // MARK: - BPE byte_fallback scoping

    /// Upstream BPE (see `BPE::tokenize_with_cache` in
    /// `tokenizers/src/models/bpe/model.rs`) only emits byte-fallback tokens
    /// (`<0xXX>`) when `byte_fallback` is set on the model. With the flag `false`
    /// (upstream default), OOV pieces fall back to the unknown token instead.
    @Test
    func bpeByteFallbackFalseFallsBackToUnk() async throws {
        // Minimal BPE: no merges, vocab contains `<unk>` and the hex codes for
        // `a` / `b`. With `byte_fallback: false`, tokenize("ab") must NOT emit the
        // hex codes even though they exist in the vocab — it emits `<unk>` twice.
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0, "<0x61>": 1, "<0x62>": 2 },
                "merges": [],
                "unk_token": "<unk>",
                "byte_fallback": false
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["<unk>", "<unk>"])
    }

    /// With `byte_fallback: true` and every byte's `<0xXX>` entry present in the
    /// vocab, OOV pieces emit byte tokens. Matches upstream `BPE::tokenize_with_cache`.
    @Test
    func bpeByteFallbackTrueEmitsHex() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0, "<0x61>": 1, "<0x62>": 2 },
                "merges": [],
                "unk_token": "<unk>",
                "byte_fallback": true
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["<0x61>", "<0x62>"])
    }

    /// With `byte_fallback: true` but the byte hex codes missing from vocab, OOV
    /// pieces fall back to the unknown token. Mirrors upstream's all-or-nothing
    /// byte fallback: `tokens: Option<Vec<_>> = s.bytes().map(... vocab.get).collect()`
    /// in `BPE::tokenize_with_cache` returns `None` if any byte is missing, and
    /// falls through to the unk-token branch.
    @Test
    func bpeByteFallbackTrueButMissingHexFallsBackToUnk() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "<unk>", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "BPE",
                "vocab": { "<unk>": 0 },
                "merges": [],
                "unk_token": "<unk>",
                "byte_fallback": true
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: "ab") == ["<unk>", "<unk>"])
    }

    // MARK: - Added-tokens phase-2 normalization

    /// Phase-2 matches `normalized: true` added tokens against the normalized input.
    /// Upstream's `AddedVocabulary::refresh_added_tokens` (in
    /// `tokenizers/src/tokenizer/added_vocabulary.rs`) populates its `normalized_cache`
    /// by running the configured normalizer over each token's content, so the
    /// normalized pattern must equal the normalized input to match. An added token
    /// `<Day>` with `normalized: true` under a `Lowercase` normalizer should match
    /// input `"a <Day>"` because both sides lowercase to `<day>`.
    @Test
    func addedTokenNormalizedMatchesAfterNormalizerTransform() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<Day>", "special": false, "normalized": true }
              ],
              "normalizer": { "type": "Lowercase" },
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<Day>": 1, "a": 2 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        // Upstream `AddedVocabulary::split_with_indices` stores the normalized
        // slice (here `"<day>"`) as the Token value, so the emitted token string
        // carries the normalized form even though the encoded ID is the added
        // token's vocab-key ID.
        #expect(tokenizer.tokenize(text: "a <Day>") == ["a", "<day>"])
        #expect(tokenizer.encode(text: "a <Day>") == [2, 1])
    }

    // MARK: - single_word Unicode word-class coverage

    /// `single_word: true` must treat combining marks (Unicode category `\p{M}`)
    /// as word characters, matching the Rust `regex` crate's default `\w` used by
    /// upstream `AddedVocabulary`'s boundary regex in
    /// `tokenizers/src/tokenizer/added_vocabulary.rs`. With decomposed `"cafe\u{0301}"`
    /// immediately before `<mask>`, the combining acute `\u{0301}` is a word char,
    /// so `single_word` rejects the match and the whole string resolves via vocab.
    @Test
    func upstreamSingleWordUnicodeBoundaryDecomposed() async throws {
        let decomposedCafe = "cafe\u{0301}" // 5 scalars, `\u{0301}` is \p{M}
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": false, "single_word": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "\(decomposedCafe)<mask>": 2, "\(decomposedCafe)": 3 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        // The combining acute on the left of `<mask>` must count as a word char
        // under the Unicode `\w` definition; single_word rejects, and the full
        // string resolves as a single vocab entry.
        #expect(tokenizer.tokenize(text: "\(decomposedCafe)<mask>") == ["\(decomposedCafe)<mask>"])
    }

    // MARK: - WordPiece scalar handling

    /// `max_input_chars_per_word` counts Unicode scalars (matching upstream's
    /// `sequence.chars().count()` in `WordPiece::tokenize`, in
    /// `tokenizers/src/models/wordpiece/mod.rs`), not grapheme clusters.
    ///
    /// Fixture: `max_input_chars_per_word = 5` and an input of 3 graphemes / 6 scalars
    /// (`"a\u{0301}"` repeated 3x). The vocab contains both `"a\u{0301}"` and
    /// `"##a\u{0301}"`, so without the cut-off a subword decomposition would be
    /// emitted. Upstream rejects the input as too long and emits `[UNK]`; Swift must
    /// match.
    @Test
    func wordPieceMaxInputCharsPerWordCountsScalars() async throws {
        let aAcute = "a\u{0301}" // 1 grapheme, 2 scalars
        let tripled = String(repeating: aAcute, count: 3) // 3 graphemes, 6 scalars
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "vocab": { "[UNK]": 0, "\(aAcute)": 1, "##\(aAcute)": 2 },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 5
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: tripled) == ["[UNK]"])
    }

    /// WordPiece must walk the substring scalar-by-scalar (matching upstream's
    /// byte-offset walk with `substr.chars().last().map_or(1, |c| c.len_utf8())`
    /// inside `WordPiece::tokenize` in `tokenizers/src/models/wordpiece/mod.rs`),
    /// not grapheme-by-grapheme.
    ///
    /// Fixture: input `"e\u{0301}"` is one grapheme but two scalars; vocab has `"e"`
    /// and `"##\u{0301}"` separately. Upstream emits `["e", "##\u{0301}"]`; a
    /// grapheme-aware walk can't split inside the cluster and emits `[UNK]`.
    @Test
    func wordPieceWalksScalars() async throws {
        let combining = "e\u{0301}" // 1 grapheme, 2 scalars
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordPiece",
                "vocab": { "[UNK]": 0, "e": 1, "##\u{0301}": 2 },
                "unk_token": "[UNK]",
                "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(tokenizer.tokenize(text: combining) == ["e", "##\u{0301}"])
    }

    /// `single_word: true` must use a Unicode-aware word-boundary check. Upstream's
    /// `AddedVocabulary` uses `Regex::new(r"^\w")` / `r"\w$"` (see
    /// `tokenizers/src/tokenizer/added_vocabulary.rs`) — the Rust `regex` crate's
    /// default `\w` is Unicode's `\p{word}`, not ASCII.
    ///
    /// Fixture: input `"café<mask>"` has `<mask>` immediately after `é` (a Unicode
    /// word character). single_word must reject the match; with no match, the whole
    /// string looks up as a single vocab entry.
    @Test
    func upstreamSingleWordUnicodeBoundary() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": false, "single_word": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "café<mask>": 2, "café": 3 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        // `é` on the left of `<mask>` is a Unicode word char; single_word:true
        // rejects the match. The full string is in the vocab, so it resolves as
        // a single token.
        #expect(tokenizer.tokenize(text: "café<mask>") == ["café<mask>"])
    }

    /// Upstream reference: `single_word_tokens` in `tokenizers/tests/added_tokens.rs`.
    /// With `single_word: true`, `ing` must NOT match inside `dancing` (surrounded by
    /// word characters). With `single_word: false`, the match IS taken and splits the
    /// word.
    @Test
    func upstreamSingleWordTokens() async throws {
        let vocab = """
            "[UNK]": 0, "I": 1, "like": 2, "dancing": 3, "danc": 4, "ing": 5
            """

        // single_word: true — rejects mid-word match
        let strictJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 5, "content": "ing", "special": false, "single_word": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { \(vocab) },
                "unk_token": "[UNK]"
              }
            }
            """
        let strictDir = try Self.writeFixtureDirectory(tokenizerJSON: strictJSON)
        let strict = try await AutoTokenizer.from(directory: strictDir)
        #expect(strict.tokenize(text: "I like dancing") == ["I", "like", "dancing"])

        // single_word: false — matcher splits `dancing` into `danc` + `ing`
        let permissiveJSON = strictJSON.replacingOccurrences(
            of: "\"single_word\": true",
            with: "\"single_word\": false"
        )
        let permissiveDir = try Self.writeFixtureDirectory(tokenizerJSON: permissiveJSON)
        let permissive = try await AutoTokenizer.from(directory: permissiveDir)
        #expect(permissive.tokenize(text: "I like dancing") == ["I", "like", "danc", "ing"])
    }

    /// Upstream reference: `lstrip_tokens` in `tokenizers/tests/added_tokens.rs`.
    /// `<mask>` with `lstrip: true` consumes leading whitespace runs during matching.
    @Test
    func upstreamLstripTokens() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": true, "lstrip": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "I": 2, "saw": 3, "a": 4 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        // Whitespace run before `<mask>` is swallowed by the matcher; upstream
        // keeps it as part of the emitted Token value (see
        // `AddedVocabulary::split_with_indices` — the widened slice is stored as
        // the value), while the encoded ID is still the added-token ID.
        #expect(tokenizer.tokenize(text: "I saw a    <mask>") == ["I", "saw", "a", "    <mask>"])
    }

    /// Upstream reference: `rstrip_tokens` in `tokenizers/tests/added_tokens.rs`.
    /// `<mask>` with `rstrip: true` consumes trailing whitespace runs during matching.
    @Test
    func upstreamRstripTokens() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "<mask>", "special": true, "rstrip": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "I": 2, "saw": 3, "a": 4, "cat": 5 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        // Trailing whitespace is consumed by the `rstrip` match and retained on
        // the emitted Token value (widened-slice rule), matching upstream.
        #expect(tokenizer.tokenize(text: "I saw a <mask>    cat") == ["I", "saw", "a", "<mask>    ", "cat"])
    }

    /// Upstream reference: `overlapping_tokens` in `tokenizers/tests/added_tokens.rs`.
    /// With added tokens `danc`, `nci`, `ing` on input `"I like dancing"`:
    /// - upstream (Aho-Corasick) picks `danc` (4 chars) over `nci` (3 chars) at
    ///   position 7, then `ing` at position 11; `nci` never fires.
    /// - Our NSRegex alternation pre-sorted by length desc gives the same behavior.
    @Test
    func upstreamOverlappingTokens() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "danc", "special": false, "normalized": false },
                { "id": 2, "content": "nci", "special": false, "normalized": false },
                { "id": 3, "content": "ing", "special": false, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "danc": 1, "nci": 2, "ing": 3, "I": 4, "like": 5 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        #expect(tokenizer.tokenize(text: "I like dancing") == ["I", "like", "danc", "ing"])
    }

    /// Upstream reference: the second half of `overlapping_tokens` in
    /// `tokenizers/tests/added_tokens.rs`. The insertion order of added tokens must
    /// not change the match result; the matcher's structure is order-independent.
    @Test
    func upstreamOverlappingTokensInsertionOrderIndependent() async throws {
        // Identical to `upstreamOverlappingTokens` but with added_tokens listed in a
        // shuffled order and an additional `ike` token that splits `like`.
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                { "id": 1, "content": "nci", "special": false, "normalized": false },
                { "id": 2, "content": "danc", "special": false, "normalized": false },
                { "id": 3, "content": "ing", "special": false, "normalized": false },
                { "id": 4, "content": "ike", "special": false, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "nci": 1, "danc": 2, "ing": 3, "ike": 4, "I": 5, "l": 6 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        // `ike` matches at pos 3, `danc` at pos 7, `ing` at pos 11; `nci` never
        // fires. Unmatched `"I l"` pre-tokenizes to `["I", "l"]`.
        #expect(tokenizer.tokenize(text: "I like dancing") == ["I", "l", "ike", "danc", "ing"])
    }

    /// Upstream reference: `tokenizers/src/tokenizer/added_vocabulary.rs`
    /// `test_lstrip_unicode_space`. A single `<mask>` with `lstrip+rstrip+single_word`
    /// matches around a regular space, a tab (`\t`), and the Mongolian vowel separator
    /// (`\u{2000}`, which is whitespace but not a word character).
    ///
    /// Uses `pre_tokenizer: null` so the unmatched chunks between `<mask>` matches
    /// go to the model step whole. This ensures the assertion exercises the added-
    /// tokens matcher: if it rejected any `<mask>` match, the surrounding span would
    /// reach the model as one chunk and resolve to `[UNK]` instead of splitting into
    /// the expected `hi`, `<mask>`, `there`, `<mask>`, `<mask>` sequence.
    @Test
    func upstreamLstripUnicodeSpace() async throws {
        let tokenizerJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false },
                {
                  "id": 1, "content": "<mask>", "special": false,
                  "lstrip": true, "rstrip": true, "single_word": true,
                  "normalized": false
                }
              ],
              "normalizer": { "type": "Lowercase" },
              "pre_tokenizer": null,
              "post_processor": null,
              "decoder": null,
              "model": {
                "type": "WordLevel",
                "vocab": { "[UNK]": 0, "<mask>": 1, "hi": 2, "there": 3 },
                "unk_token": "[UNK]"
              }
            }
            """
        let directory = try Self.writeFixtureDirectory(tokenizerJSON: tokenizerJSON)
        let tokenizer = try await AutoTokenizer.from(directory: directory)

        let tokens = tokenizer.tokenize(text: "Hi <mask> there\t<mask>\t<mask>\u{2000}")
        // Each `<mask>` match is widened by its `lstrip+rstrip` flags, so the
        // surrounding whitespace (regular space, tab, and the Mongolian vowel
        // separator) stays on the emitted token values — mirroring upstream's
        // `AddedVocabulary::split_with_indices` storing the widened slice.
        #expect(tokens == ["hi", " <mask> ", "there", "\t<mask>\t", "<mask>\u{2000}"])
    }

    /// Upstream reference: `pipeline_bert` in `tokenizers/tests/documentation.rs`.
    /// A WordPiece model decodes with `##` prefixes visible when no decoder is
    /// configured (tokens joined with spaces), and decodes as concatenated subwords
    /// when a `WordPiece` decoder is attached.
    @Test
    func upstreamWordPieceDecoderOnAndOff() async throws {
        let wordPieceModelBlock: String = """
              "type": "WordPiece",
              "vocab": {
                "[UNK]": 0,
                "welcome": 1,
                "to": 2,
                "the": 3,
                "tok": 4,
                "##eni": 5,
                "##zer": 6,
                "##s": 7,
                "library": 8
              },
              "unk_token": "[UNK]",
              "continuing_subword_prefix": "##",
              "max_input_chars_per_word": 100
            """

        // No decoder — subwords surface with `##` prefix and tokens are
        // space-joined.
        let noDecoderJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": null,
              "model": {
            \(wordPieceModelBlock)
              }
            }
            """
        let noDecoderDir = try Self.writeFixtureDirectory(
            tokenizerJSON: noDecoderJSON,
            tokenizerConfigJSON: #"{ "clean_up_tokenization_spaces": false }"#
        )
        let noDecoder = try await AutoTokenizer.from(directory: noDecoderDir)
        let ids = noDecoder.encode(text: "welcome to the tokenizers library", addSpecialTokens: false)
        let noDecoderOut = noDecoder.decode(tokenIds: ids, skipSpecialTokens: true)
        #expect(noDecoderOut == "welcome to the tok ##eni ##zer ##s library")

        // With WordPiece decoder — subwords are concatenated, `##` is stripped,
        // separator tokens join with a single space.
        let withDecoderJSON: String = """
            {
              "version": "1.0",
              "added_tokens": [
                { "id": 0, "content": "[UNK]", "special": true, "normalized": false }
              ],
              "normalizer": null,
              "pre_tokenizer": { "type": "Whitespace" },
              "post_processor": null,
              "decoder": { "type": "WordPiece", "prefix": "##", "cleanup": false },
              "model": {
            \(wordPieceModelBlock)
              }
            }
            """
        let withDecoderDir = try Self.writeFixtureDirectory(
            tokenizerJSON: withDecoderJSON,
            tokenizerConfigJSON: #"{ "clean_up_tokenization_spaces": false }"#
        )
        let withDecoder = try await AutoTokenizer.from(directory: withDecoderDir)
        let ids2 = withDecoder.encode(text: "welcome to the tokenizers library", addSpecialTokens: false)
        let withDecoderOut = withDecoder.decode(tokenIds: ids2, skipSpecialTokens: true)
        #expect(withDecoderOut == "welcome to the tokenizers library")
    }
    #endif
}

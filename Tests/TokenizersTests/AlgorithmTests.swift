// Copyright © Anthony DePasquale

import Foundation
import Testing

@testable import Tokenizers

/// Tests for algorithm-level behavior exercised through tiny on-disk fixtures. These
/// cover cases the hub-integration suite can't reach on its own: the WordLevel
/// decode-fallback path and Rust-backed Unigram behavior.
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
        let encoded = try tokenizer.encode(text: "The quick brown fox jumps over the lazy dog")
        #expect(encoded == expected)

        let encoding = try tokenizer.encodeWithMetadata(text: "The quick brown fox jumps over the lazy dog")
        #expect(encoding.tokenIds == expected)
        #expect(encoding.tokens.count == encoding.tokenIds.count)
        #expect(encoding.attentionMask == Array(repeating: 1, count: expected.count))
        #expect(encoding.specialTokensMask == Array(repeating: 0, count: expected.count))
        #expect(encoding.sequenceCount == 1)
        #expect(encoding.offsetUnit == .unicodeScalar)

        // No decoder in the fixture -> upstream falls back to joining with " ". The
        // fixture also sets `clean_up_tokenization_spaces: false` so the space-join
        // survives the backend's cleanup pass.
        let decoded = try tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false)
        #expect(decoded == "the quick brown fox jumps over the lazy dog")

        // Out-of-vocabulary words fall through to the unknown token.
        #expect(try tokenizer.encode(text: "Hello unknown world") == [1, 0, 2])
    }

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
        // `ab` (score -1.0) beats `a` + `b` (-3.0 - 3.0 = -6.0).
        let directory = try Self.unigramFixture(vocab: [
            ("<unk>", -20.0),
            ("a", -3.0),
            ("b", -3.0),
            ("ab", -1.0),
        ])
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        #expect(try tokenizer.tokenize(text: "ab") == ["ab"])
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
        #expect(try tokenizer.tokenize(text: "ab") == ["a", "b"])
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
        // "hello" can split as he+ll+o (-3) or hell+o (-6) or he+l+l+o (fails because there is no "l"). The algorithm picks he+ll+o since -3 > -6.
        #expect(try tokenizer.tokenize(text: "hello") == ["he", "ll", "o"])
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
        // `tokenizers/src/models/unigram/model.rs`). The Rust backend produces
        // `"xy"` as one fused unknown between the known `h` and `i`.
        #expect(try tokenizer.tokenize(text: "hxyi") == ["h", "xy", "i"])
        // `convertTokenToId` reports vocabulary membership, not tokenization
        // outcome — unknown characters are absent from the vocab and resolve to
        // `nil`, matching upstream `Tokenizer::token_to_id`.
        #expect(tokenizer.convertTokenToId("x") == nil)
        #expect(tokenizer.convertTokenToId("y") == nil)
    }
}

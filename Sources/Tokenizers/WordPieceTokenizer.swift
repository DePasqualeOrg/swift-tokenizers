#if Swift
// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import TokenizersCore

/// A WordPiece model step, matching upstream `tokenizers/src/models/wordpiece/mod.rs`.
///
/// Implements only the model step of the pipeline: vocabulary lookup with greedy
/// longest-prefix subword matching. Normalization (e.g. BERT lowercasing, CJK spacing)
/// and pre-tokenization are applied by the backend's `Normalizer` / `PreTokenizer`
/// factories from `tokenizer.json`, not by this type.
final class WordPieceTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    private let tokensToIds: [String: Int]
    private let idsToTokens: [Int: String]

    private let unkToken: String
    private let continuingSubwordPrefix: String
    private let maxInputCharsPerWord: Int

    let bosToken: String?
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    let unknownToken: String?
    let unknownTokenId: Int?

    /// Upstream WordPiece (`tokenizers/src/models/wordpiece/mod.rs`) has no `fuse_unk`
    /// field — it's a Unigram-only concept. Hardcoded false so a stray `fuse_unk`
    /// in `tokenizer_config.json` can't accidentally fold consecutive `[UNK]`s.
    let fuseUnknownTokens: Bool = false

    required init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        vocab: TokenizerVocab? = nil,
        merges: TokenizerMerges? = nil
    ) throws {
        guard let configVocab = tokenizerData.model.vocab.dictionary() else {
            throw TokenizerError.missingVocab
        }
        var tokensToIds: [String: Int] = [:]
        tokensToIds.reserveCapacity(configVocab.count + addedTokens.count)
        for (key, value) in configVocab {
            if let id = value.integer() {
                tokensToIds[key.string] = id
            }
        }

        for (token, id) in addedTokens {
            tokensToIds[token] = id
        }

        self.tokensToIds = tokensToIds
        self.idsToTokens = tokensToIds.reduce(into: [:]) { result, element in
            result[element.value] = element.key
        }

        self.unkToken = tokenizerData.model["unkToken"].string() ?? "[UNK]"
        self.continuingSubwordPrefix = tokenizerData.model["continuingSubwordPrefix"].string() ?? "##"
        self.maxInputCharsPerWord = tokenizerData.model["maxInputCharsPerWord"].integer() ?? 100

        // Upstream `WordPiece::tokenize` in `tokenizers/src/models/wordpiece/mod.rs`
        // errors at tokenize time if the unk token isn't in vocab. Our non-throwing
        // tokenize signature can't bubble that, so we fail here — silent nil would
        // crash later inside `encode`'s force-unwrap.
        guard let unknownTokenId = tokensToIds[unkToken] else {
            throw TokenizerError.missingUnknownToken(model: "WordPiece")
        }
        self.unknownToken = unkToken
        self.unknownTokenId = unknownTokenId

        // If `bosToken` / `eosToken` are declared but not in vocab, the ID stays
        // `nil` silently — matching the cross-backend Rust `resolve_token_id` fallback.
        self.bosToken = tokenizerConfig.bosToken.tokenString
        self.bosTokenId = bosToken.flatMap { tokensToIds[$0] }
        self.eosToken = tokenizerConfig.eosToken.tokenString
        self.eosTokenId = eosToken.flatMap { tokensToIds[$0] }
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token] ?? unknownTokenId
    }

    func convertIdToToken(_ id: Int) -> String? {
        idsToTokens[id]
    }

    /// Tokenizes a single pre-tokenized word. The caller is responsible for running
    /// normalization and pre-tokenization first; this method receives one word at a time.
    ///
    /// Walks the input scalar-by-scalar to match upstream `WordPiece::tokenize` in
    /// `tokenizers/src/models/wordpiece/mod.rs`, which operates on byte offsets and
    /// decrements by the last scalar's UTF-8 length. Swift's grapheme-cluster-level
    /// `String.Index` would refuse to split inside a combining-mark cluster,
    /// producing different tokenization on non-ASCII input.
    func tokenize(text: String) -> [String] {
        let scalars = text.unicodeScalars
        if scalars.count > maxInputCharsPerWord {
            return [unkToken]
        }

        var subTokens: [String] = []
        var start = scalars.startIndex
        let end = scalars.endIndex

        while start < end {
            var cursor = end
            var matched: String?
            var matchedEnd = start
            while start < cursor {
                var candidate = String(scalars[start..<cursor])
                if start > scalars.startIndex {
                    candidate = continuingSubwordPrefix + candidate
                }
                if tokensToIds[candidate] != nil {
                    matched = candidate
                    matchedEnd = cursor
                    break
                }
                cursor = scalars.index(before: cursor)
            }
            guard let matched else {
                return [unkToken]
            }
            subTokens.append(matched)
            start = matchedEnd
        }

        return subTokens
    }
}
#endif

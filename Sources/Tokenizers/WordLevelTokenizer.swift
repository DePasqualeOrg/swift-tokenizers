#if Swift
// Copyright © Anthony DePasquale

import Foundation
import TokenizersCore

/// A WordLevel model step, matching upstream `tokenizers/src/models/wordlevel/mod.rs`.
///
/// Vocabulary lookup with an unknown-token fallback; no subword decomposition. The
/// caller is responsible for normalization and pre-tokenization (applied by the
/// backend's `Normalizer` / `PreTokenizer` factories from `tokenizer.json`).
final class WordLevelTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    private let tokensToIds: [String: Int]
    private let idsToTokens: [Int: String]

    let bosToken: String?
    let bosTokenId: Int?
    let eosToken: String?
    let eosTokenId: Int?
    let unknownToken: String?
    let unknownTokenId: Int?

    /// Upstream WordLevel (`tokenizers/src/models/wordlevel/mod.rs`) has no `fuse_unk`
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

        // Resolve the model's unknown token from `tokenizer.json.model.unk_token`,
        // defaulting to `"<unk>"` per upstream `WordLevel` (`wordlevel/mod.rs`).
        // The user-facing unknown token (which may instead come from
        // `tokenizer_config.json.unk_token`) is composed at the tokenizer layer by
        // `PreTrainedTokenizer.unknownToken`, matching the sidecar resolution order.
        let unk = tokenizerData.model["unkToken"].string() ?? "<unk>"
        // Upstream `WordLevel::tokenize` errors at tokenize time if the unk token
        // isn't in vocab. Our non-throwing tokenize signature can't bubble that,
        // so we fail here instead of letting the caller observe a silent empty
        // tokenization or a later force-unwrap crash.
        guard let unknownTokenId = tokensToIds[unk] else {
            throw TokenizerError.missingUnknownToken(model: "WordLevel")
        }
        self.unknownToken = unk
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

    func tokenize(text: String) -> [String] {
        if tokensToIds[text] != nil {
            return [text]
        }
        if let unknownToken {
            return [unknownToken]
        }
        return []
    }
}
#endif

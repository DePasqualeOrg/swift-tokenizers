#if TOKENIZERS_SWIFT_BACKEND
// Copyright © Hugging Face SAS
// Copyright © Anthony DePasquale

import Foundation
import TokenizersCore

private struct ImmutableBox<Value>: @unchecked Sendable {
    let value: Value
}

/// A Unigram tokenizer implementation based on the SentencePiece algorithm.
///
/// Unigram tokenizers use a probabilistic approach where each token has a score,
/// and the tokenization process finds the most probable segmentation of the input text.
/// This is commonly used in models like T5 and XLM-RoBERTa.
class UnigramTokenizer: PreTrainedTokenizerModel, @unchecked Sendable {
    /// A token with its associated score in the Unigram model.
    struct SentencePieceToken: Sendable {
        var token: String
        var score: Float
    }

    /// The complete vocabulary of tokens with their scores.
    let vocab: [SentencePieceToken]

    /// The special token used for unknown/out-of-vocabulary text.
    let unknownPiece: SentencePieceToken

    /// The score associated with the unknown token.
    var unknownTokenScore: Float { unknownPiece.score }

    /// The numeric ID of the unknown token.
    let unknownTokenId: Int?

    /// The unknown token string.
    var unknownToken: String? { unknownPiece.token }

    /// The minimum score found in the vocabulary (used for score calculations).
    let minScore: Float

    /// Mapping from token strings to their numeric IDs.
    let tokensToIds: [NSString: Int]

    /// The beginning-of-sequence token, read from `tokenizer_config.json` like the
    /// other models. Unigram upstream has no model-level bos; the lattice's sentinel
    /// id is tracked separately via `latticeBosTokenId`.
    let bosToken: String?

    /// The numeric ID of the beginning-of-sequence token, resolved through the
    /// vocabulary.
    let bosTokenId: Int?

    /// Sentinel token id used to seed the Viterbi lattice. Any value works because
    /// the BOS lattice node has length 0 and contributes no score; we just need a
    /// stable placeholder.
    private let latticeBosTokenId: Int

    /// The end-of-sequence token string, if defined.
    let eosToken: String?

    /// The numeric ID of the end-of-sequence token, if defined.
    let eosTokenId: Int?

    /// Whether consecutive unknown tokens should be fused. Upstream's `Unigram::from`
    /// in `tokenizers/src/models/unigram/model.rs` defaults this to `true` when
    /// constructing the model; the flag lives on `tokenizer.json`'s `model.fuse_unk`,
    /// not in `tokenizer_config.json`.
    let fuseUnknownTokens: Bool

    private let trie: Trie<Character>

    // MARK: - Static Builder Methods

    /// Parses raw JSON vocab data into SentencePieceToken array.
    ///
    /// - Parameter rawVocab: Array of [token, score] pairs from JSON
    /// - Returns: Parsed vocabulary array
    static func buildVocab(from rawVocab: [[Any]]) -> [SentencePieceToken] {
        rawVocab.compactMap { pair -> SentencePieceToken? in
            guard pair.count == 2,
                let token = pair[0] as? String
            else { return nil }

            let score: Float
            if let floatScore = pair[1] as? Double {
                score = Float(floatScore)
            } else if let intScore = pair[1] as? Int {
                score = Float(intScore)
            } else {
                return nil
            }

            return SentencePieceToken(token: token, score: score)
        }
    }

    /// Builds the token-to-ID mapping dictionary from a vocab array.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: Dictionary mapping token strings (as NSString) to their integer IDs
    static func buildTokensToIds(from vocab: [SentencePieceToken]) -> [NSString: Int] {
        Dictionary(
            uniqueKeysWithValues: vocab.map { $0.token as NSString }.enumerated().map { ($1, $0) }
        )
    }

    fileprivate static func buildTokensToIdsBox(
        from vocab: [SentencePieceToken]
    ) -> ImmutableBox<[NSString: Int]> {
        ImmutableBox(value: buildTokensToIds(from: vocab))
    }

    /// Builds a character trie from the vocabulary for prefix matching.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: A populated Trie for efficient common-prefix search
    static func buildTrie(from vocab: [SentencePieceToken]) -> Trie<Character> {
        let trie = Trie<Character>()
        trie.append(contentsOf: vocab.map { $0.token })
        return trie
    }

    /// Computes the minimum score across all tokens in the vocabulary.
    ///
    /// - Parameter vocab: The parsed vocabulary array
    /// - Returns: The minimum score value
    static func computeMinScore(from vocab: [SentencePieceToken]) -> Float {
        vocab.reduce(999) { partial, token in
            min(partial, token.score)
        }
    }

    // MARK: - Private Designated Initializer

    /// Initializer accepting pre-built data structures.
    /// All other initializers and the async factory delegate to this one.
    private init(
        vocab: [SentencePieceToken],
        tokensToIds: [NSString: Int],
        trie: Trie<Character>,
        minScore: Float,
        unknownTokenId: Int,
        fuseUnknownTokens: Bool,
        tokenizerConfig: Config
    ) {
        self.vocab = vocab
        self.tokensToIds = tokensToIds
        self.trie = trie
        self.minScore = minScore
        self.unknownTokenId = unknownTokenId
        self.fuseUnknownTokens = fuseUnknownTokens
        self.unknownPiece = SentencePieceToken(
            token: vocab[unknownTokenId].token,
            score: minScore - 10
        )

        let bos = tokenizerConfig.bosToken.tokenString
        self.bosToken = bos
        self.bosTokenId = bos.flatMap { tokensToIds[$0 as NSString] }
        // The lattice only needs a stable placeholder id for the length-0 BOS
        // node, so fall back to 0 when no vocab entry resolves.
        self.latticeBosTokenId = self.bosTokenId ?? 0

        let eos = tokenizerConfig.eosToken.string()
        self.eosToken = eos
        if let eos {
            self.eosTokenId = tokensToIds[eos as NSString]
        } else {
            self.eosTokenId = nil
        }
    }

    // MARK: - Protocol Conformance Init

    /// Initializes a Unigram tokenizer from configuration data with optional pre-extracted vocab.
    ///
    /// When vocab is provided (fast path), builds data structures from raw arrays.
    /// Otherwise falls back to parsing from Config.
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data containing vocabulary and scores
    ///   - addedTokens: Additional tokens to include in the vocabulary
    ///   - vocab: Pre-extracted vocabulary (nil to parse from Config)
    ///   - merges: Pre-extracted merge rules (unused by Unigram)
    /// - Throws: `TokenizerError` if the vocabulary is missing or malformed
    required convenience init(
        tokenizerConfig: Config,
        tokenizerData: Config,
        addedTokens: [String: Int],
        vocab: TokenizerVocab? = nil,
        merges: TokenizerMerges? = nil
    ) throws {
        let vocabArray: [SentencePieceToken]

        if case .unigram(let rawVocabArray) = vocab, let rawVocab = rawVocabArray as? [[Any]] {
            // Fast path: build from pre-extracted raw data
            vocabArray = Self.buildVocab(from: rawVocab)
        } else {
            // Fallback: parse from Config
            guard let configVocab = tokenizerData.model.vocab.array() else {
                throw TokenizerError.missingVocab
            }

            vocabArray = try configVocab.map { piece -> SentencePieceToken in
                let tuple = piece.array(or: [])

                guard let token = tuple.first?.string(),
                    let scoreValue = tuple.last
                else {
                    throw TokenizerError.malformedVocab
                }

                let score: Float
                if let floatScore = scoreValue.floating() {
                    score = floatScore
                } else if let numberScore = scoreValue.integer() {
                    score = Float(numberScore)
                } else {
                    throw TokenizerError.malformedVocab
                }

                return SentencePieceToken(token: token, score: score)
            }
        }

        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else {
            throw TokenizerError.malformedVocab
        }
        // Upstream raises `UnigramError::MissingUnkId` at tokenize time. Our
        // non-throwing tokenize signature can't bubble, and the designated init
        // accesses `vocab[unknownTokenId]` — turn a silent index-out-of-range crash
        // into a clear error.
        guard unknownTokenId < vocabArray.count else {
            throw TokenizerError.missingUnknownToken(model: "Unigram")
        }

        self.init(
            vocab: vocabArray,
            tokensToIds: Self.buildTokensToIds(from: vocabArray),
            trie: Self.buildTrie(from: vocabArray),
            minScore: Self.computeMinScore(from: vocabArray),
            unknownTokenId: unknownTokenId,
            fuseUnknownTokens: tokenizerData.model["fuseUnk"].boolean(or: true),
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Async Factory

    /// Async factory that builds data structures in parallel for faster loading.
    ///
    /// Phase 1: Parse raw vocab array into [SentencePieceToken] (sequential, since later
    ///          phases all depend on the parsed vocab).
    /// Phase 2: Build tokensToIds, trie, and computeMinScore in parallel (independent).
    ///
    /// - Parameters:
    ///   - tokenizerConfig: The tokenizer configuration
    ///   - tokenizerData: The tokenizer data (used for unkId)
    ///   - rawVocab: Raw vocab array of [token, score] pairs
    ///   - addedTokens: Additional tokens to include in the vocabulary
    /// - Returns: A fully initialized UnigramTokenizer
    /// - Throws: `TokenizerError` if the vocabulary is malformed
    static func create(
        tokenizerConfig: Config,
        tokenizerData: Config,
        rawVocab: [[Any]],
        addedTokens: [String: Int]
    ) async throws -> UnigramTokenizer {
        guard let unknownTokenId = tokenizerData.model["unkId"].integer() else {
            throw TokenizerError.malformedVocab
        }

        // Phase 1: Parse raw vocab (sequential, all other phases depend on this)
        let vocabArray = buildVocab(from: rawVocab)
        guard unknownTokenId < vocabArray.count else {
            throw TokenizerError.missingUnknownToken(model: "Unigram")
        }

        // Phase 2: Build independent data structures in parallel
        async let tokensToIdsTask = buildTokensToIdsBox(from: vocabArray)
        async let trieTask = buildTrie(from: vocabArray)
        async let minScoreTask = computeMinScore(from: vocabArray)

        let tokensToIds = await tokensToIdsTask
        let trie = await trieTask
        let minScore = await minScoreTask

        return UnigramTokenizer(
            vocab: vocabArray,
            tokensToIds: tokensToIds.value,
            trie: trie,
            minScore: minScore,
            unknownTokenId: unknownTokenId,
            fuseUnknownTokens: tokenizerData.model["fuseUnk"].boolean(or: true),
            tokenizerConfig: tokenizerConfig
        )
    }

    // MARK: - Token Conversion

    /// Converts a token string to its corresponding numeric ID.
    ///
    /// - Parameter token: The token string to convert
    /// - Returns: The numeric ID, or the unknown token ID if not found
    func convertTokenToId(_ token: String) -> Int? {
        tokensToIds[token as NSString] ?? unknownTokenId
    }

    /// Converts a numeric token ID back to its string representation.
    ///
    /// - Parameter id: The numeric token ID to convert
    /// - Returns: The token string
    func convertIdToToken(_ id: Int) -> String? {
        vocab[id].token
    }

    // MARK: - Tokenization

    /// Tokenizes input text using the Unigram algorithm with dynamic programming.
    ///
    /// - Parameter text: The input text to tokenize
    /// - Returns: An array of token strings representing the most probable segmentation
    func tokenize(text: String) -> [String] {
        var lattice = TokenLattice(sentence: text, bosTokenId: latticeBosTokenId, eosTokenId: eosTokenId ?? 0)

        // Populate nodes
        let sentence = lattice.sentence
        var beginPos = 0
        while beginPos < sentence.count {
            let mblen = 1
            var hasSingleNode = false

            let beginIndex = sentence.index(sentence.startIndex, offsetBy: beginPos)
            for token in trie.commonPrefixSearchIterator(sentence[beginIndex...]).map({ String($0) }) {
                // The trie is built from the same vocab as `tokensToIds`, so a miss
                // here indicates a corrupted tokenizer; skip the node and let the
                // unknown fallback below cover the position.
                guard let tokenId = tokensToIds[token as NSString] else { continue }
                let tokenScore = vocab[tokenId].score
                lattice.insert(startOffset: beginPos, length: token.count, score: tokenScore, tokenId: tokenId)
                if !hasSingleNode, token.count == mblen {
                    hasSingleNode = true
                }
            }
            if !hasSingleNode {
                lattice.insert(startOffset: beginPos, length: mblen, score: unknownTokenScore, tokenId: unknownTokenId ?? 0)
            }
            beginPos += mblen
        }

        // Upstream `Unigram::tokenize` (`tokenizers/src/models/unigram/model.rs`)
        // fuses consecutive unknown nodes by accumulating their raw text spans
        // and flushing them as a single token on the next non-unknown boundary.
        // Each unknown node covers exactly one character, so string concatenation
        // of the spans matches upstream's byte-span concatenation.
        let viterbiNodes = lattice.viterbi()
        var tokens: [String] = []
        tokens.reserveCapacity(viterbiNodes.count)
        var previousWasUnknown = false
        for node in viterbiNodes {
            let piece = String(lattice.piece(node))
            let isUnknown = node.tokenId == unknownTokenId
            if fuseUnknownTokens, isUnknown, previousWasUnknown, let last = tokens.last {
                tokens[tokens.count - 1] = last + piece
            } else {
                tokens.append(piece)
            }
            previousWasUnknown = isUnknown
        }
        return tokens
    }
}
#endif
